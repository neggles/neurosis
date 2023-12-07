from typing import Optional

import torch
from torch import Tensor, nn

from .embedding import AbstractEmbModel


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
        uc = {self.input_key: uc.long()}
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
