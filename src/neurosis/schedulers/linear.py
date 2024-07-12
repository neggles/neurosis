import logging

import numpy as np

from neurosis.utils import ensure_list

logger = logging.getLogger(__name__)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(
        self,
        warm_up_steps: int | list[int],
        f_min: float | list[float],
        f_max: float | list[float],
        f_start: float | list[float],
        cycle_lengths: int | list[int],
        verbosity_interval=0,
    ):
        self.lr_warm_up_steps = ensure_list(warm_up_steps)
        self.f_start = ensure_list(f_start)
        self.f_min = ensure_list(f_min)
        self.f_max = ensure_list(f_max)
        self.cycle_lengths = ensure_list(cycle_lengths)

        list_len = len(self.lr_warm_up_steps)
        if not all(len(x) == list_len for x in [self.f_min, self.f_max, self.f_start, self.cycle_lengths]):
            raise ValueError("all argument lists must have the same length")

        self.total_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cycle_len in self.total_cycles[1:]:
            if n <= cycle_len:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.total_cycles[cycle]
        if (self.verbosity_interval > 0) and (n % self.verbosity_interval == 0):
            logger.info(f"current step: {n}, last lr-multiplier: {self.last_f}, " f"current cycle {cycle}")

        f_min = self.f_min[cycle]
        f_max = self.f_max[cycle]
        f_start = self.f_start[cycle]
        cycle_length = self.cycle_lengths[cycle]
        lr_warm_up_steps = self.lr_warm_up_steps[cycle]

        if n < lr_warm_up_steps:
            f = (f_max - f_start) / lr_warm_up_steps * n + f_start
            self.last_f = f
            return f
        else:
            t = min((n - lr_warm_up_steps) / (cycle_length - lr_warm_up_steps), 1.0)
            f = f_min + 0.5 * (f_max - f_min) * (1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.total_cycles[cycle]
        if (self.verbosity_interval > 0) and (n % self.verbosity_interval == 0):
            logger.info(f"current step: {n}, last lr-multiplier: {self.last_f}, " f"current cycle {cycle}")

        f_min = self.f_min[cycle]
        f_max = self.f_max[cycle]
        f_start = self.f_start[cycle]
        cycle_length = self.cycle_lengths[cycle]
        lr_warm_up_steps = self.lr_warm_up_steps[cycle]

        if n < self.lr_warm_up_steps[cycle]:
            f = (f_max - f_start) / lr_warm_up_steps * n + f_start
            self.last_f = f
            return f
        else:
            f = f_min + (f_max - f_min) * (cycle_length - n) / (cycle_length)
            self.last_f = f
            return f
