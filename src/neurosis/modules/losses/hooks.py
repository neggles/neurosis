from abc import ABC, abstractmethod

import lightning.pytorch as L
from torch import Tensor


class LossHook(ABC):
    """A hook for the loss function in DiffusionEngine that modifies the loss before it is backpropagated."""

    def __init__(self, name: str = "GenericHook", **kwargs):
        self.name = name

    @abstractmethod
    def __call__(
        self,
        engine: L.LightningModule,
        batch: dict,
        loss: Tensor,
        loss_dict: dict[str, Tensor] = {},
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError("You called an ABC. Why would you do that? ;_;")
