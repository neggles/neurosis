import logging
from typing import Any, Optional, Union

import kornia
import numpy as np
import open_clip
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip import CLIPTextConfig
from transformers.tokenization_utils import BatchEncoding

from neurosis.modules.encoders.embedding import AbstractEmbModel
from neurosis.utils import autocast, expand_dims_like, np_text_decode, silence_hf_load_warnings

logger = logging.getLogger(__name__)


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
        extended_chunks: int = 0,
        load_pretrained: bool = False,
        **kwargs,
    ):
        # clip-vit-base-patch32
        super().__init__(**kwargs)
        if layer not in self.LAYERS:
            raise ValueError(f"layer must be one of {self.LAYERS}, got {layer=}")

        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version)
        with silence_hf_load_warnings():
            if load_pretrained:
                logger.debug(f"Loading pretrained model weights from {version}")
                self.transformer: CLIPTextModel = CLIPTextModel.from_pretrained(version)
            else:
                logger.debug(f"Loading CLIP config from {version} but skipping weights")
                clip_config = CLIPTextConfig.from_pretrained(version)
                self.transformer: CLIPTextModel = CLIPTextModel(clip_config)

        self.device = device
        self.max_length = max_length
        if not self.is_trainable:
            self.freeze()

        self.layer = layer
        self.return_pooled = always_return_pooled
        self.output_hidden_states = self.layer in ["hidden", "penultimate"]
        self.extended_chunks = extended_chunks
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

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
                self.layer_idx = 10
            case _:
                raise ValueError("Only last, penultimate and hidden layers are supported")

    def freeze(self):
        self.transformer = self.transformer.eval()
        super().freeze()

    def forward(self, text: Union[str, list[str]]):
        # decode any numpy bytearrays and ensure text is a list
        text = np_text_decode(text, aslist=True)

        # hijack the uncond rate for empty-prompt dropout
        if self.ucg_rate > 0.0 and self.ucg_rate < np.random.rand():
            text = [""] * len(text)

        if self.extended_chunks > 1:
            # extended mode
            z: list[Tensor] = []
            pooled: list[Tensor] = []

            batch_tokens: Tensor = self.tokenize_extended(text)["input_ids"]  # Batch, Chunk, Tokens

            # encode each chunk as if it were a minibatch
            for sample in batch_tokens:
                outputs: BaseModelOutputWithPooling = self.transformer(
                    input_ids=sample, output_hidden_states=self.output_hidden_states
                )
                # get appropriate layer output
                match self.layer:
                    case "last":
                        z.append(outputs.last_hidden_state.unsqueeze(0))
                    case "pooled":
                        z.append(outputs.pooler_output[:, None, :].unsqueeze(0))
                    case _:
                        z.append(outputs.hidden_states[self.layer_idx + 1].unsqueeze(0))
                # save the first pooled embedding, but only the first
                if self.return_pooled:
                    pooled.append(outputs.pooler_output[0])
            # concat on channel dim
            z: Tensor = torch.stack(z, dim=0)
            z = z.reshape(z.shape[0], -1, z[0].shape[-1])
            if self.return_pooled:
                return z, pooled
            else:
                return z

        else:
            batch_tokens: Tensor = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].to(self.device)

            outputs: BaseModelOutputWithPooling = self.transformer(
                input_ids=batch_tokens, output_hidden_states=self.output_hidden_states
            )
            match self.layer:
                case "last":
                    z = outputs.last_hidden_state
                case "pooled":
                    z = outputs.pooler_output[:, None, :]
                case _:
                    z = outputs.hidden_states[self.layer_idx + 1]

            if self.return_pooled:
                return z, outputs.pooler_output
            return z

    def encode(self, text):
        return self(text)

    def tokenize(self, text: list[str]) -> BatchEncoding:
        input_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(self.device)
        return BatchEncoding({"input_ids": input_ids})

    def tokenize_extended(self, text: list[str]) -> BatchEncoding:
        chunk_tokens = self.tokenizer.model_max_length - 2  # -2 for start and end tokens
        max_tokens = self.extended_chunks * chunk_tokens
        n_prompts = len(text)

        # get the input_ids for the prompt (sans special tokens)
        input_ids: Tensor = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=False,
            max_length=max_tokens,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(self.device)

        # reshape into chunks
        input_ids = input_ids.view(n_prompts, self.extended_chunks, chunk_tokens)
        bos_eos_shape = input_ids.shape[:2] + (1,)
        # prepend BOS token and append EOS token to each chunk
        input_ids = torch.cat(
            (
                torch.full(bos_eos_shape, self.tokenizer.bos_token_id, dtype=torch.long, device=self.device),
                input_ids,
                torch.full(bos_eos_shape, self.tokenizer.eos_token_id, dtype=torch.long, device=self.device),
            ),
            dim=2,
        )

        return BatchEncoding({"input_ids": input_ids})


OPENCLIP_2_MAP = {
    "default": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "ViT-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
}


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text, with pooling layer.
    Uses legacy unconditioned value dropout.
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch: str = "ViT-bigG-14",
        version: Optional[str] = "laion2b_s39b_b160k",
        device: Union[str, torch.device] = "cuda",
        max_length: int = 77,
        layer: str = "last",
        always_return_pooled: bool = False,
        legacy: bool = False,
        extended_chunks: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if layer not in self.LAYERS:
            raise ValueError(f"layer must be one of {self.LAYERS}, got {layer=}")

        with silence_hf_load_warnings():
            model, _, _ = open_clip.create_model_and_transforms(
                arch, device=torch.device("cpu"), pretrained=version
            )
        del model.visual
        self.model: open_clip.CLIP = model

        tokenizer_repo = OPENCLIP_2_MAP.get(arch, None)
        if tokenizer_repo is None:
            logger.warning(f"Could not find tokenizer for {arch=} and {version=}, using default")
            tokenizer_repo = OPENCLIP_2_MAP["default"]
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_repo)

        self.device = torch.device(device)
        self.max_length = max_length
        if not self.is_trainable:
            self.freeze()

        self.layer = layer
        self.return_pooled = always_return_pooled
        self.legacy = legacy

        self.extended_chunks = extended_chunks
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.embed_dim = self.model.text_projection.shape[-1]

        if self.return_pooled and self.legacy:
            raise ValueError("legacy mode does not support returning pooled embeddings!")
        if extended_chunks > 1 and self.legacy:
            raise ValueError("legacy mode does not support extended chunks!")

    def freeze(self) -> None:
        self.model = self.model.eval()
        super().freeze()

    def forward(self, text: Union[str, list[str]]) -> Tensor | tuple[Tensor, Tensor]:
        text = np_text_decode(text, aslist=True)

        # hijack the uncond rate for empty-prompt dropout
        if self.ucg_rate > 0.0 and self.ucg_rate < np.random.rand():
            text = [""] * len(text)

        if self.extended_chunks > 1:
            # extended mode
            z = []
            pooled = []

            batch_tokens: Tensor = self.tokenize_extended(text)["input_ids"]  # Batch, Chunk, Tokens

            # encode each chunk as if it were a minibatch
            for sample in batch_tokens:
                outputs = self.encode_with_transformer(sample)
                # save the first pooled embedding, but only the first
                if self.return_pooled:
                    pooled.append(outputs["pooled"][0:1])
                # append the layer output to the list
                z.append(outputs[self.layer].unsqueeze(0))

            # for pooled, stack the pooled embeddings for the batch
            if self.return_pooled:
                pooled = torch.cat(pooled, dim=0)

            # concat on channel dim
            z = torch.stack(z, dim=0)
            z = z.reshape(z.shape[0], -1, z.shape[-1])
            if self.return_pooled:
                return z, pooled
            else:
                return z

        else:
            # standard mode
            batch_tokens: Tensor = self.tokenize(text)["input_ids"]  # Batch, Tokens
            z = self.encode_with_transformer(batch_tokens)
            if self.legacy:
                return z
            if self.return_pooled:
                return z[self.layer], z["pooled"]
            else:
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

    def encode(self, text: str | list[str]) -> Tensor:
        return self.forward(text)

    def tokenize(self, text: list[str]) -> BatchEncoding:
        input_ids = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(self.device)
        return BatchEncoding({"input_ids": input_ids})

    def tokenize_extended(self, text: list[str]) -> BatchEncoding:
        chunk_tokens = self.tokenizer.model_max_length - 2  # -2 for start and end tokens
        max_tokens = self.extended_chunks * chunk_tokens
        n_prompts = len(text)

        # get the input_ids for the prompt (sans special tokens)
        input_ids: Tensor = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=False,
            max_length=max_tokens,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(self.device)

        # reshape into chunks
        input_ids = input_ids.view(n_prompts, self.extended_chunks, chunk_tokens)
        bos_eos_shape = input_ids.shape[:2] + (1,)
        # prepend BOS token and append EOS token to each chunk
        input_ids = torch.cat(
            (
                torch.full(bos_eos_shape, self.tokenizer.bos_token_id, dtype=torch.long, device=self.device),
                input_ids,
                torch.full(bos_eos_shape, self.tokenizer.eos_token_id, dtype=torch.long, device=self.device),
            ),
            dim=2,
        )

        return BatchEncoding({"input_ids": input_ids})


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

        with silence_hf_load_warnings():
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
                logger.warn(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"I hope you know what you're doing!."
                )

        return x, tokens if self.output_tokens else x

    def encode(self, text: str) -> Tensor:
        return self.forward(text)
