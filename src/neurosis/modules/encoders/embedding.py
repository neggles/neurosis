import logging
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.utils import count_params, np_text_decode

logger = logging.getLogger(__name__)


class AbstractEmbModel(nn.Module):
    name: Optional[str]
    input_key: Optional[str]
    input_keys: Optional[list[str]]
    ucg_rate: float
    is_trainable: bool
    base_lr: Optional[float]

    def __init__(
        self,
        name: Optional[str] = None,
        input_key: Optional[str] = None,
        ucg_rate: Optional[float] = 0.0,
        is_trainable: Optional[bool] = None,
        base_lr: Optional[float] = None,
    ):
        super().__init__()
        if not hasattr(self, "is_trainable"):
            self.is_trainable = is_trainable or False
        if not hasattr(self, "ucg_rate"):
            self.ucg_rate = ucg_rate
        if not hasattr(self, "input_key") and input_key is not None:
            self.input_key = input_key
        if not hasattr(self, "base_lr") and base_lr is not None:
            self.base_lr = base_lr

        if not hasattr(self, "name"):
            self.name = name or str(self.__class__.__name__)

    def freeze(self) -> None:
        # set self to eval mode
        self.eval()
        # set requires_grad to False for all parameters
        self.requires_grad_(False)

    @property
    def context(self):
        if self.is_trainable:
            return nullcontext
        return torch.no_grad


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

            embedders.append(embedder)

        if len(embedders) == 0:
            raise ValueError("no embedders were added! what is my purpose? why am I here? check your config!")

        self.embedders: list[AbstractEmbModel] = nn.ModuleList(embedders)
        self.rng = np.random.default_rng()

    def forward(
        self,
        batch: dict[str, Tensor | str | np.bytes_, np.ndarray],
        force_zero_embeddings: Optional[list] = None,
    ) -> dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []

        embedder: AbstractEmbModel
        for embedder in self.embedders:
            with embedder.context():
                if getattr(embedder, "input_key", None) is not None:
                    inputs = batch[embedder.input_key]
                    if isinstance(inputs, list) and isinstance(inputs[0], (str, np.bytes_, np.ndarray)):
                        inputs = np_text_decode(inputs, aslist=True)
                    elif isinstance(inputs, (str, np.bytes_, np.ndarray)):
                        inputs = np_text_decode(inputs)
                    elif (
                        isinstance(inputs, list) and embedder.__class__.__name__ == "ConcatTimestepEmbedderND"
                    ):
                        inputs = torch.tensor(
                            inputs, device=batch["image"].device, dtype=batch["image"].dtype
                        )

                    if embedder.ucg_rate > 0.0 and embedder.input_key == "caption":
                        if self.rng.random() < embedder.ucg_rate:
                            inputs = [" "] * len(inputs)

                    emb_out = embedder(inputs)

                elif getattr(embedder, "input_keys", None) is not None:
                    inputs = [batch[k] for k in embedder.input_keys]
                    for idx in range(len(inputs)):
                        if isinstance(inputs[idx][0], (str, np.bytes_, np.ndarray)):
                            inputs[idx] = np_text_decode(inputs[idx], aslist=True)
                    emb_out = embedder(*inputs)

            if not isinstance(emb_out, (Tensor, list, tuple)):
                raise ValueError(f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}")

            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]

            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                elif embedder.ucg_rate > 0.0 and embedder.input_key != "caption":
                    emb = emb.mul(
                        torch.bernoulli(
                            torch.full((emb.shape[0],), 1.0 - embedder.ucg_rate, device=emb.device)
                        ).reshape((-1,) + (1,) * (emb.dim() - 1))
                    )

                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        return output

    @contextmanager
    def zero_ucg(self):
        rates = []
        shared_ucg = self.shared_ucg
        self.shared_ucg = 0.0
        for embedder in self.embedders:
            rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        try:
            yield None
        finally:
            self.shared_ucg = shared_ucg
            for embedder, rate in zip(self.embedders, rates):
                embedder.ucg_rate = rate

    def get_unconditional_conditioning(
        self,
        batch_c: dict,
        batch_uc: Optional[dict] = None,
        force_uc_zero_embeddings: Optional[list[str]] = None,
        force_cond_zero_embeddings: Optional[list[str]] = None,
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []

        with self.zero_ucg():
            c = self(batch_c, force_zero_embeddings=force_cond_zero_embeddings)
            if batch_uc is None:
                batch_uc = batch_c.copy()
                batch_uc["caption"] = ([""] * len(batch_c["caption"])) if "caption" in batch_c else [""]
            uc = self(batch_uc, force_zero_embeddings=force_uc_zero_embeddings)

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
