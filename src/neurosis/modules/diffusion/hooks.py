import logging
from abc import ABC, abstractmethod
from typing import Optional

import lightning.pytorch as L
from torch import Tensor

logger = logging.getLogger(__name__)


class LossHook(ABC):
    """A hook for the loss function in DiffusionEngine that modifies the loss before it is backpropagated."""

    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs,
    ):
        self.name = name or self.__class__.__name__
        if len(kwargs) > 0:
            logger.info(f"{self.__class__.__name__} superclass received unexpected kwargs:\n" + f"{kwargs}")

    def __call__(
        self,
        pl_module: L.LightningModule,
        batch: dict,
        loss: Tensor,
        loss_dict: dict[str, Tensor] = {},
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.batch_hook(pl_module, batch, loss, loss_dict, **kwargs)

    def pre_hook(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx,
    ):
        return batch

    @abstractmethod
    def batch_hook(
        self,
        pl_module: L.LightningModule,
        batch: dict,
        loss: Tensor,
        loss_dict: dict[str, Tensor] = {},
        **kwargs,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError("Abstract base class was called ;_;")
