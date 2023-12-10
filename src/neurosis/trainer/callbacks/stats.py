import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor

if TYPE_CHECKING:
    from torch.cuda import _CudaDeviceProperties


logger = logging.getLogger(__name__)


class CallbackInterval(str, Enum):
    GlobalStep = "global_step"
    Epoch = "epoch"
    Batch = "batch"


class GPUMemoryUsage(Callback):
    def __init__(
        self,
        disabled: bool = False,
        every_n_steps: int = 1,
        step_type: CallbackInterval = CallbackInterval.GlobalStep,
        before_batch: bool = False,
        after_batch: bool = True,
    ):
        super().__init__()
        if before_batch and after_batch:
            raise ValueError("before_batch and after_batch cannot both be True")

        self.enabled = not disabled
        self.every_n = every_n_steps
        self.step_type = step_type
        self.before_batch = before_batch
        self.after_batch = after_batch

        self.device: torch.device = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # save our device
        device = pl_module.device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if self.device.type == "cuda":
            self.device_props: "_CudaDeviceProperties" = torch.cuda.get_device_properties(self.device)
            self.total_memory = self.device_props.total_memory
        else:
            self.device_props = {}
            self.total_memory = -1

    def __get_step(self, trainer: Trainer, batch_idx: int) -> int:
        if self.step_type == CallbackInterval.GlobalStep:
            return trainer.global_step
        elif self.step_type == CallbackInterval.Batch:
            return batch_idx
        elif self.step_type == CallbackInterval.Epoch:
            return trainer.current_epoch
        else:
            raise ValueError(f"Unknown step type {self.step_type}")

    def check_interval(self, trainer: Trainer, batch_idx: int) -> bool:
        if self.enabled:
            try:
                step_num = self.__get_step(trainer, batch_idx)
                return (step_num % self.every_n) == 0
            except ValueError:
                logger.warning(f"Unknown step type: {self.step_type}! Disabling callback.")
                self.enabled = False
        return False

    def get_memory_usage(self, prefix: Optional[str] = None):
        if not torch.cuda.is_available():
            return {"index": -1, "total": -1, "free": -1}
        memory_stats = torch.cuda.memory_stats_as_nested_dict(self.device)
        stats_dict = {
            "index": self.device.index,
            "total": self.total_memory,
            "free": self.total_memory - memory_stats["allocated_bytes"]["all"]["current"],
            "free_min": self.total_memory - memory_stats["allocated_bytes"]["all"]["peak"],
            "reserved": memory_stats["reserved_bytes"]["all"]["current"],
            "reserved_max": memory_stats["reserved_bytes"]["all"]["peak"],
            "used": memory_stats["allocated_bytes"]["all"]["current"],
            "used_max": memory_stats["allocated_bytes"]["all"]["peak"],
        }
        if prefix is not None:
            return {f"{prefix}/{k}": v for k, v in stats_dict.items()}
        return stats_dict

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        if not (self.before_batch and self.check_interval(trainer, batch_idx)):
            return  # skip
        pl_module.log_dict(
            self.get_memory_usage("gpu/memory"), prog_bar=False, logger=True, on_step=True, on_epoch=True
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Union[Tensor, dict[str, Any]],
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        if not (self.after_batch and self.check_interval(trainer, batch_idx)):
            return  # skip
        pl_module.log_dict(
            self.get_memory_usage("gpu/memory"), prog_bar=False, logger=True, on_step=True, on_epoch=True
        )
