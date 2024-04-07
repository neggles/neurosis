import logging
from abc import ABC, abstractmethod
from warnings import warn

from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class AbstractLRSchedule(ABC):
    last_epoch: int

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            logger.warning(f"{self.__class__.__name__}: ignored extra args: {args}")
        if len(kwargs) > 0:
            logger.warning(f"{self.__class__.__name__}: ignored extra kwargs: {kwargs}")

    def __call__(self, step: int, **kwargs):
        return self.schedule(step, **kwargs)

    @abstractmethod
    def schedule(self, step: int, **kwargs) -> float | Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")


class BaseLRScheduler(LambdaLR):
    _get_lr_called_within_step: bool

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

        if len(schedules) != num_groups:
            raise ValueError(f"expected 1 or {num_groups} schedules, got {len(schedules)}")
        else:
            logger.info(f"using {len(schedule)} schedules for {num_groups} param groups")
            schedules = list(schedule)

        super().__init__(optimizer, schedules, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")
        return [sched(self.last_epoch) for sched in self.lr_lambdas]
