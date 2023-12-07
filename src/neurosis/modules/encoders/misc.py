from torch import Tensor

from .embedding import AbstractEmbModel


class IdentityEncoder(AbstractEmbModel):
    def encode(self, x: Tensor) -> Tensor:
        return x

    def forward(self, x: Tensor) -> Tensor:
        return x
