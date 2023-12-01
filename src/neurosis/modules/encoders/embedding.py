import logging
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.modules.diffusion.model import Encoder
from neurosis.modules.regularizers import DiagonalGaussianDistribution
from neurosis.utils import count_params, disabled_train, expand_dims_like
from neurosis.utils.module import extract_into_tensor, make_beta_schedule

logger = logging.getLogger(__name__)


class AbstractEmbModel(nn.Module):
    input_keys: Optional[List[str]]
    legacy_ucg_val: Optional[float]
    ucg_prng: Optional[np.random.RandomState]

    def __init__(
        self,
        is_trainable: Optional[bool] = None,
        ucg_rate: Optional[float] = 0.0,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        if not hasattr(self, "is_trainable") and is_trainable is not None:
            self.is_trainable = is_trainable
        if not hasattr(self, "ucg_rate") and ucg_rate is not None:
            self.ucg_rate = ucg_rate
        if not hasattr(self, "input_key") and input_key is not None:
            self.input_key = input_key

        if hasattr(self, "legacy_ucg_val") and self.legacy_ucg_val is not None:
            self.ucg_prng = np.random.RandomState()
        else:
            self.ucg_prng = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @is_trainable.deleter
    def is_trainable(self) -> None:
        del self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, Tensor]:
        return self._ucg_rate

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, Tensor]):
        self._ucg_rate = value

    @ucg_rate.deleter
    def ucg_rate(self) -> None:
        del self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @input_key.deleter
    def input_key(self) -> None:
        del self._input_key

    def freeze(self) -> None:
        # set self to eval mode
        self.eval()
        # set requires_grad to False for all parameters
        self.requires_grad_(False)
        # set train method to disabled_train
        self.train = disabled_train


class ClassEmbedder(AbstractEmbModel):
    def __init__(
        self,
        embed_dim,
        n_classes=1000,
        add_sequence_dim=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c: Tensor) -> Tensor:
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc


class ClassEmbedderForMultiCond(ClassEmbedder):
    def forward(self, batch: Tensor, key: Optional[str] = None, disable_dropout: bool = False) -> Tensor:
        out = batch
        key = key or self.key
        islist = isinstance(batch[key], list)
        if islist:
            batch[key] = batch[key][0]
        c_out = super().forward(batch, key, disable_dropout)
        out[key] = [c_out] if islist else c_out
        return out


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(
        self,
        emb_models: list[AbstractEmbModel],
    ):
        super().__init__()
        embedders: list[AbstractEmbModel] = []
        for idx, embedder in enumerate(emb_models):
            emb_class = embedder.__class__.__name__
            if not isinstance(embedder, AbstractEmbModel):
                raise ValueError(f"embedder model #{idx} {emb_class} is not a subclass of AbstractEmbModel")

            logger.info(
                f"Initialized embedder #{idx}: {emb_class} "
                + f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if not any((hasattr(embedder, "input_key"), hasattr(embedder, "input_keys"))):
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder #{idx} {emb_class}")

            embedder.legacy_ucg_val = getattr(embedder, "legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)

        if len(embedders) == 0:
            raise ValueError("no embedders were added! what is my purpose? why am I here? check your config!")

        self.embedders = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: dict) -> dict:
        if embedder.legacy_ucg_val is None:
            raise ValueError("embedder has no legacy_ucg_val")

        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def forward(self, batch: dict, force_zero_embeddings: Optional[List] = None) -> dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []

        embedder: AbstractEmbModel
        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    if embedder.legacy_ucg_val is not None:
                        batch = self.possibly_get_ucg_val(embedder, batch)
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])

            if not isinstance(emb_out, (Tensor, list, tuple)):
                raise ValueError(f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}")

            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]
            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x


class SpatialRescaler(nn.Module):
    def __init__(
        self,
        n_stages: int = 1,
        method: str = "bilinear",
        multiplier: float = 0.5,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        bias: bool = False,
        wrap_video: bool = False,
        kernel_size: int = 1,
        remap_output: bool = False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in [
            "nearest",
            "linear",
            "bilinear",
            "trilinear",
            "bicubic",
            "area",
        ]
        self.multiplier = multiplier
        self.interpolator = partial(F.interpolate, mode=method)
        self.remap_output = out_channels is not None or remap_output
        if self.remap_output:
            logger.info(
                f"Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing."
            )
            self.channel_mapper = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=kernel_size // 2,
            )
        self.wrap_video = wrap_video

    def forward(self, x: Tensor):
        if self.wrap_video and x.ndim == 5:
            B, C, T, H, W = x.shape
            x = rearrange(x, "b c t h w -> b t c h w")
            x = rearrange(x, "b t c h w -> (b t) c h w")

        for _ in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.wrap_video:
            x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T, c=C)
            x = rearrange(x, "b t c h w -> b c t h w")
        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x: Tensor):
        return self(x)


class LowScaleEncoder(nn.Module):
    # type annotations for the buffers
    betas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    log_one_minus_alphas_cumprod: Tensor
    sqrt_recip_alphas_cumprod: Tensor
    sqrt_recipm1_alphas_cumprod: Tensor

    def __init__(
        self,
        model: Encoder,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        timesteps: int = 1000,
        max_noise_level: int = 250,
        output_size: int = 64,
        scale_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_noise_level = max_noise_level
        self.model = model
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise: Optional[Tensor] = None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, x: Tensor):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = F.interpolate(z, size=self.out_size, mode="nearest")
        # z = z.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return z, noise_level

    def decode(self, z: Tensor):
        z = z / self.scale_factor
        return self.model.decode(z)
