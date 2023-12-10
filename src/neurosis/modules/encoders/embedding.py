import logging
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.utils import count_params, disabled_train, expand_dims_like

logger = logging.getLogger(__name__)


class AbstractEmbModel(nn.Module):
    input_key: Optional[str]
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
        if not hasattr(self, "is_trainable"):
            self.is_trainable = is_trainable or False
        if not hasattr(self, "ucg_rate"):
            self.ucg_rate = ucg_rate or 0.0
        if not hasattr(self, "input_key") and input_key is not None:
            self.input_key = input_key

        # synchronize ucg_rate and ucg_prng for legacy ucg mode
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

        self.embedders: nn.ModuleList[AbstractEmbModel] = nn.ModuleList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: dict) -> tuple[dict]:
        if embedder.legacy_ucg_val is None:
            raise ValueError("embedder has no legacy_ucg_val")

        is_ucg = [False] * len(batch[embedder.input_key])
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
                is_ucg[i] = True
        batch["is_ucg"] = is_ucg
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

    def get_unconditional_conditioning(
        self,
        batch_c: dict,
        batch_uc: Optional[dict] = None,
        force_uc_zero_embeddings: Optional[list[str]] = None,
        force_cond_zero_embeddings: Optional[list[str]] = None,
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


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
