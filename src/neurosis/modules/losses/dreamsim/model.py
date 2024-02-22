from abc import abstractmethod

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor
from torch.nn import functional as F
from torchvision.transforms import v2 as T

from .common import ensure_tuple
from .vit import VisionTransformer, vit_base_dreamsim


class DreamsimBackbone(ModelMixin, ConfigMixin):
    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor:
        raise NotImplementedError("abstract base class was called ;_;")

    def forward(self, x: Tensor) -> Tensor:
        """Dreamsim forward pass for similarity computation.
        Args:
            x (Tensor): Input tensor of shape [2, B, 3, H, W].

        Returns:
            sim (torch.Tensor): dreamsim similarity score of shape [B].
        """
        inputs = x.view(-1, 3, *x.shape[-2:])

        x = self.forward_features(inputs).view(*x.shape[:2], -1)

        return 1 - F.cosine_similarity(x[0], x[1], dim=1)


class DreamsimModel(DreamsimBackbone):
    @register_to_config
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        pre_norm: bool = False,
        act_layer: str = "gelu",
        img_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        img_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        do_resize: bool = False,
    ) -> None:
        super().__init__()

        self.image_size = ensure_tuple(image_size, 2)
        self.patch_size = ensure_tuple(patch_size, 2)
        self.layer_norm_eps = layer_norm_eps
        self.pre_norm = pre_norm
        self.do_resize = do_resize
        self.img_mean = img_mean
        self.img_std = img_std

        num_classes = 512 if self.pre_norm else 0
        self.extractor: VisionTransformer = vit_base_dreamsim(
            image_size=image_size,
            patch_size=patch_size,
            layer_norm_eps=layer_norm_eps,
            num_classes=num_classes,
            pre_norm=pre_norm,
            act_layer=act_layer,
        )

        self.resize = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.img_norm = T.Normalize(mean=self.img_mean, std=self.img_std)

        self._compiled = False

    def transforms(self, x: Tensor) -> Tensor:
        if self.do_resize:
            x = self.resize(x)
        return self.img_norm(x)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.transforms(x)
        x = self.extractor(x, norm=self.pre_norm)

        x = x.div(x.norm(dim=1, keepdim=True))
        x = x.sub(x.mean(dim=1, keepdim=True))
        return x


class DreamsimEnsemble(DreamsimBackbone):
    @register_to_config
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float | tuple[float, ...] = (1e-6, 1e-5, 1e-5),
        num_classes: int | tuple[int, ...] = (0, 512, 512),
        do_resize: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(layer_norm_eps, float):
            layer_norm_eps = (layer_norm_eps,) * 3
        if isinstance(num_classes, int):
            num_classes = (num_classes,) * 3

        self.image_size = ensure_tuple(image_size, 2)
        self.patch_size = ensure_tuple(patch_size, 2)
        self.do_resize = do_resize

        self.dino: VisionTransformer = vit_base_dreamsim(
            image_size=self.image_size,
            patch_size=self.patch_size,
            layer_norm_eps=layer_norm_eps[0],
            num_classes=num_classes[0],
            pre_norm=False,
            act_layer="gelu",
        )
        self.clip1: VisionTransformer = vit_base_dreamsim(
            image_size=self.image_size,
            patch_size=self.patch_size,
            layer_norm_eps=layer_norm_eps[1],
            num_classes=num_classes[1],
            pre_norm=True,
            act_layer="quick_gelu",
        )
        self.clip2: VisionTransformer = vit_base_dreamsim(
            image_size=self.image_size,
            patch_size=self.patch_size,
            layer_norm_eps=layer_norm_eps[2],
            num_classes=num_classes[2],
            pre_norm=True,
            act_layer="gelu",
        )

        self.resize = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.dino_norm = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.clip_norm = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        self._compiled = False

    def transforms(self, x: Tensor, resize: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        if resize:
            x = self.resize(x)
        x = self.dino_norm(x), self.clip_norm(x), self.clip_norm(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x_dino, x_clip1, x_clip2 = self.transforms(x, self.do_resize)

        # these expect to always receive a batch, and will return a batch
        x_dino = self.dino(x_dino, norm=False)
        x_clip1 = self.clip1(x_clip1, norm=True)
        x_clip2 = self.clip2(x_clip2, norm=True)

        z: Tensor = torch.cat([x_dino, x_clip1, x_clip2], dim=1)
        z = z.div(z.norm(dim=1, keepdim=True))
        z = z.sub(z.mean(dim=1, keepdim=True))
        return z
