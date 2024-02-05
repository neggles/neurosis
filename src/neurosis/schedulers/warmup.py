import logging
from bisect import bisect_left
from typing import Optional

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .base import AbstractLRSchedule

logger = logging.getLogger(__name__)


class CosineDecayWithWarmup(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,  # number of steps to warm up before hitting max_lr
        decay_steps: int,  # number of steps to decay before hitting min_lr
        base_lr: float | list[float] = 1e-6,
        max_lr: float | list[float] = 1e-3,
        min_lr: Optional[float | list[float]] = None,
        last_epoch: int = -1,
        verbose: bool = False,
        step_interval: int = 1,  # used when accumulating gradients to make math easier
    ):
        if min_lr is None:
            min_lr = base_lr

        self.warmup_steps = warmup_steps // step_interval
        self.decay_steps = decay_steps // step_interval
        self.total_steps = self.warmup_steps + self.decay_steps
        self.base_lrs: list[float] = []
        self.max_lrs: list[float] = []
        self.min_lrs: list[float] = []

        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
        self.init_lr(base_lr, max_lr, min_lr)

    def init_lr(self, base_lr: float, max_lr: float, min_lr: float):
        self.base_lrs = [base_lr] * len(self.optimizer.param_groups)
        self.max_lrs = [max_lr] * len(self.optimizer.param_groups)
        self.min_lrs = [min_lr] * len(self.optimizer.param_groups)

        if len(self.base_lrs) != len(self.max_lrs) != len(self.min_lrs):
            raise ValueError("init, max, and min LRs must be lists of the same length, or a float")

        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr

    def get_lr(self):
        if self.last_epoch == 0:
            logger.info(f"initializing learning rates: {self.base_lrs}, {self.max_lrs}, {self.min_lrs}")
            # step zero aka initial LR
            return [lr for lr, _ in zip(self.base_lrs, self.optimizer.param_groups)]

        elif self.last_epoch > self.total_steps:
            # constant min_lr after decay
            return [min_lr for min_lr, _ in zip(self.min_lrs, self.optimizer.param_groups)]

        elif self.last_epoch < self.warmup_steps:
            # linear warmup phase
            return [
                (max_lr - base_lr) * self.last_epoch / self.warmup_steps + base_lr
                for base_lr, max_lr, _ in zip(self.base_lrs, self.max_lrs, self.optimizer.param_groups)
            ]

        else:
            # cosine decay phase
            t = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                min_lr + (max_lr - min_lr) * (1 + np.cos(t * np.pi)) / 2.0
                for min_lr, max_lr, _ in zip(self.min_lrs, self.max_lrs, self.optimizer.param_groups)
            ]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class CosineWarmupSchedule(AbstractLRSchedule):
    """
    note: use with a base_lr of 1.0
    """

    def __init__(
        self,
        warm_up_steps: int,
        max_decay_steps: int,
        lr_min: float | list[float],
        lr_max: float | list[float],
        lr_start: Optional[float | list[float]] = None,
        verbose: bool = False,
    ):
        self.warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.max_decay_steps = max_decay_steps
        self.verbose = verbose

        self.last_lr = 0.0

    def schedule(self, n: int, **kwargs) -> float:
        if n < self.warm_up_steps:
            # linear warmup
            lr = (self.lr_max - self.lr_start) / self.warm_up_steps * n + self.lr_start
        else:
            # cosine decay
            t = (n - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps)
            t = min(t, 1.0)  # clamp to 1.0
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))

        self.last_lr = lr
        return lr


class CosineWarmupStagedSchedule(AbstractLRSchedule):
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(
        self,
        warm_up_steps: list[int],
        f_min: list[float],
        f_max: list[float],
        f_start: list[float],
        cycle_lengths: list[int],
        verbose: bool = False,
        verbose_interval: int = 0,
    ):
        if not all((isinstance(x, list) for x in (warm_up_steps, f_min, f_max, f_start, cycle_lengths))):
            raise ValueError("all frequency stages must be lists")

        if not all([len(x) == len(warm_up_steps) for x in (f_min, f_max, f_start, cycle_lengths)]):
            raise ValueError("all stage lists must have the same length")

        self.warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))

    def schedule(self, n: int, **kwargs):
        cycle = bisect_left(self.cum_cycles[1:], n)
        n = n - self.cum_cycles[cycle]

        if self.verbose and (n % self.verbose_interval == 0):
            logger.info(f"current step: {n}, last lr mult: {self.last_f}, cycle #{cycle:03d}")

        if n < self.warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.warm_up_steps[cycle] * n + self.f_start[
                cycle
            ]
        else:
            t = (n - self.warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))

        self.last_f = f
        return f


class LinearWarmupSchedule(CosineWarmupStagedSchedule):
    def schedule(self, n: int, **kwargs):
        cycle = bisect_left(self.cum_cycles[1:], n)
        n = n - self.cum_cycles[cycle]

        if self.verbose and (n % self.verbose_interval == 0):
            logger.info(f"current step: {n}, last lr mult: {self.last_f}, cycle #{cycle:03d}")

        if n < self.warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.warm_up_steps[cycle] * n + self.f_start[
                cycle
            ]
        else:
            f = (
                self.f_min[cycle]
                + (self.f_max[cycle] - self.f_min[cycle])
                * (self.cycle_lengths[cycle] - n)
                / (self.cycle_lengths[cycle])
            )

        self.last_f = f
        return f
