import logging
from functools import lru_cache
from time import perf_counter
from typing import Optional

from lightning.pytorch import Callback, LightningModule, Trainer
from pynvml import (
    nvmlDeviceGetComputeRunningProcesses,
    nvmlDeviceGetHandleByIndex,
    nvmlInit,
    nvmlShutdown,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def get_device_handle(index: int):
    """Cache the device handle to avoid repeated lookups."""
    return nvmlDeviceGetHandleByIndex(index)


class ConflictAbortCallback(Callback):
    """Callback which checks if other processes have started on the same GPU and aborts if so."""

    def __init__(
        self,
        enabled: bool = True,
        interval: float = 5.0,
        max_others: int = 0,
    ):
        super().__init__()
        self.enabled = enabled
        self.interval = interval
        self.max_others = max_others
        if max_others < 0:
            raise ValueError("max_others must be a non-negative integer!")

        self.last_check = 0.0
        self.stopped_epoch = 0
        self.stopped_step = 0

        self._pl_module: Optional[LightningModule] = None
        self._trainer: Optional[Trainer] = None
        self._stage: Optional[str] = None
        self._nvml_ready = False

    @property
    def max_proc(self):
        return self.max_others + 1

    def nvml_init(self):
        if not self._nvml_ready:
            logger.debug("Initializing NVML")
            get_device_handle.cache_clear()
            nvmlInit()
            self._nvml_ready = True

    def nvml_shutdown(self):
        if self._nvml_ready:
            logger.debug("Releasing NVML")
            get_device_handle.cache_clear()
            nvmlShutdown()
            self._nvml_ready = False

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "interval": self.interval,
            "max_others": self.max_others,
            "stopped_epoch": self.stopped_epoch,
            "stopped_step": self.stopped_step,
        }

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        logger.debug(f"ConflictAbortCallback setting up for {stage}")
        self._pl_module = pl_module
        self.last_check = 0.0
        self.nvml_init()
        logger.info(f"ConflictAbortCallback ready for {stage}")

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        logger.debug(f"ConflictAbortCallback teardown after {stage}")
        self.nvml_shutdown()

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if pl_module.device.type != "cuda":
            return  # this callback only works on nVidia GPUs for the moment
        if self.should_check():
            self.check_gpu_conflict(trainer, pl_module)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx: int
    ) -> None:
        if pl_module.device.type != "cuda":
            return  # this callback only works on nVidia GPUs for the moment
        if self.should_check():
            self.check_gpu_conflict(trainer, pl_module)

    def should_check(self) -> bool:
        return perf_counter() - self.last_check > self.interval

    def get_device_processes(self, index: int) -> list:
        device_handle = get_device_handle(index)
        return nvmlDeviceGetComputeRunningProcesses(device_handle)

    def check_gpu_conflict(self, trainer: Trainer, pl_module: LightningModule):
        gpu_id = pl_module.device.index
        if gpu_id is None:
            return  # no GPU to check

        processes = self.get_device_processes(gpu_id)
        n_proc = len(processes)
        should_stop = n_proc > self.max_proc

        # synchronize all processes to ensure that if one process is stopping, all are
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop

        if trainer.should_stop:
            logger.exception(
                f"ConflictAbortCallback: {n_proc} processes on GPU{gpu_id} (limit {self.max_proc}), aborting!"
            )
            self.stopped_epoch = trainer.current_epoch
            self.stopped_step = trainer.global_step
            # if min_epochs or min_steps is set, that'll override should_stop, so we need to check here and raise an exception
            if (trainer.min_steps is not None and trainer.global_step < trainer.min_steps) or (
                trainer.min_epochs is not None and trainer.current_epoch < trainer.min_epochs
            ):
                raise RuntimeError("ConflictAbortCallback detected a conflict and stopped the training run")
        else:
            logger.debug(
                f"ConflictAbortCallback: {n_proc} processes on GPU{gpu_id} (limit {self.max_proc}), OK"
            )
