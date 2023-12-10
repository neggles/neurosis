from typing import Any, Optional, Union

import kornia
import open_clip
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.tokenization_utils import BatchEncoding

from neurosis.modules.encoders.embedding import AbstractEmbModel
from neurosis.utils import autocast, expand_dims_like, np_text_decode


class FrozenCLIPEmbedder(AbstractEmbModel):
    """
    Uses the CLIP transformer encoder for text (from huggingface)
    Uses legacy unconditioned value dropout.
    """

    LAYERS = ["last", "pooled", "hidden", "penultimate"]

    def __init__(
        self,
        version: str = "openai/clip-vit-large-patch14",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        layer_idx: Optional[int] = None,
        always_return_pooled: bool = False,
        **kwargs,
    ):
        # clip-vit-base-patch32
        super().__init__(**kwargs)
        if layer not in self.LAYERS:
            raise ValueError(f"layer must be one of {self.LAYERS}, got {layer=}")

        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer: CLIPTextModel = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.layer = layer
        self.return_pooled = always_return_pooled
        self.output_hidden_states = self.layer in ["hidden", "penultimate"]

        match layer:
            case "hidden":
                if layer_idx is None:
                    raise ValueError("layer_idx must be specified for hidden layer")
                if not (0 <= abs(layer_idx) <= 12):
                    raise ValueError("layer_idx must be between -12 and 12")
                # add 12 to negative indices to get the correct layer
                layer_idx += 12 if layer_idx < 0 else 0
                self.layer_idx = layer_idx
            case "penultimate":
                self.layer_idx = 11
            case _:
                raise ValueError("Only last, penultimate and hidden layers are supported")

    def freeze(self):
        self.transformer = self.transformer.eval()
        super().freeze()

    @autocast
    def forward(self, text: Union[str, list[str]]):
        text = np_text_decode(text)

        batch_encoding: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs: BaseModelOutputWithPooling = self.transformer(
            input_ids=tokens, output_hidden_states=self.output_hidden_states
        )

        match self.layer:
            case "last":
                z = outputs.last_hidden_state
            case "pooled":
                z = outputs.pooler_output[:, None, :]
            case _:
                z = outputs.hidden_states[self.layer_idx]

        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text, with no pooling
    Uses legacy unconditioned value dropout.
    """

    LAYERS = ["last", "penultimate"]

    def __init__(
        self,
        arch: str = "ViT-H-14",
        version: str = "laion2b_s32b_b79k",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layer not in self.LAYERS:
            raise ValueError(f"layer must be one of {self.LAYERS}, got {layer=}")

        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        del model.visual
        self.model = model

        self.device = torch.device(device)
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.layer = layer
        match self.layer:
            case "last":
                self.layer_idx = 0
            case "penultimate":
                self.layer_idx = 1
            case _:
                raise ValueError("Only last and penultimate layers are supported")

    def freeze(self):
        self.model = self.model.eval()
        super().freeze()

    def forward(self, text: Union[str, list[str]]) -> Tensor:
        text = np_text_decode(text)
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text: Tensor) -> Tensor:
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text: str) -> Tensor:
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text, with pooling layer.
    Uses legacy unconditioned value dropout.
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch: str = "ViT-H-14",
        version: Optional[str] = "laion2b_s32b_b79k",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        always_return_pooled: bool = False,
        legacy: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layer not in self.LAYERS:
            raise ValueError(f"layer must be one of {self.LAYERS}, got {layer=}")

        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model: open_clip.CLIP = model

        self.device = torch.device(device)
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.layer = layer
        match self.layer:
            case "last":
                self.layer_idx = 0
            case "penultimate":
                self.layer_idx = 1
            case _:
                raise ValueError("Only last and penultimate layers are supported")

        self.return_pooled = always_return_pooled
        self.legacy = legacy

    def freeze(self) -> None:
        self.model = self.model.eval()
        super().freeze()

    @autocast
    def forward(self, text: Union[str, list[str]]) -> Tensor | tuple[Tensor, Tensor]:
        text = np_text_decode(text)
        tokens: Tensor = open_clip.tokenize(text)
        z: Tensor = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            if self.legacy:
                raise ValueError("legacy mode does not support returning pooled embeddings!")
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text: Tensor) -> dict[str, Tensor]:
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, text)
            x["pooled"] = pooled
            return x

    def pool(self, x: Tensor, text: Tensor) -> Tensor:
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return x

    def text_transformer_forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text: str) -> Tensor:
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch: str = "ViT-H-14",
        version: str = "laion2b_s32b_b79k",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        freeze: bool = True,
        antialias: bool = True,
        ucg_rate: float = 0.0,
        unsqueeze_dim: bool = False,
        repeat_to_max_len: bool = False,
        num_image_crops: int = 0,
        output_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.transformer
        self.model: open_clip.CLIP = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.device = torch.device(device)
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.register_buffer("mean", Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer("std", Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x: Tensor) -> Tensor:
        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        super().freeze()

    @autocast
    def forward(self, image: Tensor, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        tokens = None
        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)
        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = torch.bernoulli((1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device))[:, None] * z
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate) * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )
        if self.unsqueeze_dim:
            z = z[:, None, :]
        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z
        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z
        elif self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                    ),
                ),
                1,
            )
            return z_pad, z_pad[:, 0, ...]
        return z

    def encode_with_vision_transformer(self, img: Tensor) -> tuple[Tensor, Any | None] | Tensor:
        if img.dim() == 5:
            if self.max_crops != img.shape[1]:
                raise ValueError("Number of crops in batch does not match max_crops")
            img = rearrange(img, "b n c h w -> (b n) c h w")
        # preprocess
        img = self.preprocess(img)
        if self.output_tokens:
            if not self.model.visual.output_tokens:
                raise ValueError("CLIP vision model has token output disabled, can't output tokens!")
            x, tokens = self.model.visual(img)
        else:
            if self.model.visual.output_tokens:
                raise ValueError("CLIP vision model has token output enabled, must output tokens!")
            x: Tensor = self.model.visual(img)
            tokens = None

        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)
            # drop out between 0 and all along the sequence axis
            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )
            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)
                print(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )

        return x, tokens if self.output_tokens else x

    def encode(self, text: str) -> Tensor:
        return self(text)
