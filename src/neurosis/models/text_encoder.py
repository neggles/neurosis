from os import PathLike
from typing import Any, Optional, Union

import open_clip
import torch
from torch import Module, Tensor
from torch.amp import autocast
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
)
from transformers.models.byt5 import ByT5Tokenizer
from transformers.models.clip import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from transformers.models.t5 import T5EncoderModel, T5Tokenizer
from transformers.tokenization_utils import BatchEncoding

from neurosis.modules.encoders import AbstractEmbModel
from neurosis.utils.module import checkpoint


class FrozenT5Embedder(AbstractEmbModel):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self,
        model_name_or_path: Union[str, PathLike] = "",
        max_length: int = 256,
        model_kwargs: dict = {},
        freeze: bool = True,
        apply_mask: bool = True,
    ):
        super().__init__()
        self.model: T5EncoderModel = T5EncoderModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, **model_kwargs)

        self.device = self.model.device
        self.dtype = self.model.dtype
        self.max_length = max_length
        self.apply_mask = apply_mask

        if freeze:
            self.freeze()

    def forward(self, text: Union[str, list[str]]) -> torch.Tensor:
        batch_encoding: BatchEncoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = batch_encoding["input_ids"].to(self.device)

        with autocast("cuda", enabled=False):
            output: BaseModelOutputWithPastAndCrossAttentions = self.model(input_ids=input_ids)
        z = output.last_hidden_state
        return z

    def encode(self, text: Union[str, list[str]]) -> torch.Tensor:
        """Encode text into a latent representation."""
        return self(text)


class FrozenByT5Embedder(AbstractEmbModel):
    """
    Uses the ByT5 transformer encoder for text. Is character-aware.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, PathLike] = "",
        max_length: int = 256,
        model_kwargs: dict = {},
        freeze: bool = True,
    ):
        super().__init__()
        self.model: T5EncoderModel = T5EncoderModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.tokenizer: ByT5Tokenizer = ByT5Tokenizer.from_pretrained(model_name_or_path, **model_kwargs)

        self.device = self.model.device
        self.dtype = self.model.dtype
        self.max_length = max_length

        if freeze:
            self.freeze()

    def forward(self, text: Union[str, list[str]]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with autocast("cuda", enabled=False):
            output: BaseModelOutputWithPastAndCrossAttentions = self.model(tokens)
        z = output.last_hidden_state
        return z

    def encode(self, text: Union[str, list[str]]) -> torch.Tensor:
        """Encode text into a latent representation."""
        return self(text)


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version: str = "openai/clip-vit-large-patch14",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        layer_idx: Optional[int] = None,
        always_return_pooled: bool = False,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer: CLIPTextModel = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    @autocast
    def forward(self, text: Union[str, list[str]]):
        batch_encoding = self.tokenizer(
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
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, outputs.pooler_output
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEmbModel):
    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch: str = "ViT-H-14",
        version: str = "laion2b_s32b_b79k",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        always_return_pooled: bool = False,
        legacy: bool = True,
    ) -> None:
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device("cpu"),
            pretrained=version,
        )
        del model.visual
        self.model: open_clip.CLIP = model

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    @autocast
    def forward(self, text: Union[str, list[str]]) -> Tensor | tuple[Tensor, Tensor]:
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]
        return z[self.layer]

    def encode_with_transformer(self, text: Tensor) -> Tensor:
        x: Tensor = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
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

    def text_transformer_forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> dict[str, Any]:
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs

    def encode(self, text):
        return self(text)
