import logging
from abc import ABC, abstractmethod

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class AbstractLRSchedule(ABC):
    @abstractmethod
    def schedule(self, n: int, **kwargs) -> float:
        raise NotImplementedError("Abstract base class was called ;_;")

    def __call__(self, n: int, **kwargs):
        return self.schedule(n, **kwargs)


class BaseLRScheduler(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        schedule: AbstractLRSchedule | list[AbstractLRSchedule],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.optimizer = optimizer
        num_groups = len(optimizer.param_groups)

        if not isinstance(schedule, (list, tuple)):
            schedules = [schedule] * num_groups
        elif len(schedule) != num_groups:
            raise ValueError(f"expected 1 or {num_groups} schedules, got {len(schedule)}")
        else:
            logger.info(f"using {len(schedule)} schedules for {num_groups} param groups")
            schedules = list(schedule)
        super().__init__(optimizer, schedules, last_epoch, verbose)
