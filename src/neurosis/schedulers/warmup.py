import logging
from bisect import bisect_left

import numpy as np

from .base import AbstractLRSchedule

logger = logging.getLogger(__name__)


class CosineWarmUpSchedule(AbstractLRSchedule):
    """
    note: use with a base_lr of 1.0
    """

    def __init__(
        self,
        warm_up_steps: int,
        lr_min: float,
        lr_max: float,
        lr_start: float,
        max_decay_steps: int,
        verbose: bool = False,
        verbose_interval: int = 0,
    ):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        self.last_lr = 0.0

    def schedule(self, n: int, **kwargs):
        if self.verbose and (n % self.verbose_interval == 0):
            logger.info(f"current step: {n}, last lr mult: {self.last_lr}")

        if n < self.lr_warm_up_steps:
            # linear warmup
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
        else:
            # cosine decay
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)  # clamp to 1.0
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))

        self.last_lr = lr
        return lr


class CosineWarmUpStagedSchedule(AbstractLRSchedule):
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

        self.lr_warm_up_steps = warm_up_steps
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

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[
                cycle
            ]
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (
                self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle]
            )
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))

        self.last_f = f
        return f


class LinearWarmUpSchedule(CosineWarmUpStagedSchedule):
    def schedule(self, n: int, **kwargs):
        cycle = bisect_left(self.cum_cycles[1:], n)
        n = n - self.cum_cycles[cycle]

        if self.verbose and (n % self.verbose_interval == 0):
            logger.info(f"current step: {n}, last lr mult: {self.last_f}, cycle #{cycle:03d}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[
                cycle
            ]
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n
            ) / (self.cycle_lengths[cycle])

        self.last_f = f
        return f
