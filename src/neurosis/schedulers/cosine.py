import logging
import warnings
from typing import Optional

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from neurosis.optimizers.types import ParamGroup

logger = logging.getLogger(__name__)


class CosineAnnealingWarmupRestarts(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: int,
        cycle_steps: int,
        lr_min: float | list[float] = 1e-6,
        lr_max: float | list[float] = 1e-3,
        lr_start: Optional[float | list[float]] = None,
        decay_factor: float = 0.9,
        last_epoch: int = -1,
        verbose: bool = False,
        verbose_interval: int = 1,
    ) -> None:
        lr_start = lr_start or lr_min

        # make these lists if they aren't already
        if isinstance(lr_min, float):
            lr_min = [lr_min] * len(optimizer.param_groups)
        if isinstance(lr_max, float):
            lr_max = [lr_max] * len(optimizer.param_groups)
        if isinstance(lr_start, float):
            lr_start = [lr_start] * len(optimizer.param_groups)

        self.lr_warm_up_steps = warm_up_steps
        self.lr_cycle_steps = cycle_steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_start = lr_start
        self.lr_decay_factor = decay_factor

        if self.lr_start[0] < self.lr_min[0]:
            raise ValueError("lr_start must be greater than or equal to lr_min")

        self.last_f = 0.0
        self.verbose = verbose
        self.verbose_interval = verbose_interval

        if self.verbose:
            logger.info(
                f"lr_warm_up_steps: {self.lr_warm_up_steps}, "
                + f"lr_cycle_steps: {self.lr_cycle_steps}, "
                + f"first_cycle_steps: {self.lr_warm_up_steps + self.lr_cycle_steps}"
            )
        super().__init__(optimizer, last_epoch, verbose)
        self._init_lrs()

    def _init_lrs(self):
        for i, param_group in enumerate(zip(self.optimizer.param_groups)):
            lr_start = self.lr_start[i]
            lr_min = self.lr_min[i]
            lr_max = self.lr_max[i]

            if self.verbose:
                logger.info(f"init grp {i}: lr_start: {lr_start}, lr_min: {lr_min}, lr_max: {lr_max}")
            param_group["lr"] = lr_start
            param_group["lr_min"] = lr_min
            param_group["lr_max"] = lr_max
            param_group["lr_start"] = lr_start

    def get_group_lr(self, group: ParamGroup, cycle_step: int, cycle_num: int) -> float:
        if cycle_num == -1:
            # linear warmup
            return (group["lr_max"] - group["lr_start"]) / (self.lr_warm_up_steps * cycle_step) + group["lr_start"]  # fmt: skip

        # set max LR by annealing decay factor (after warmup, applied per epoch)
        max_lr = group["lr_max"] * self.lr_decay_factor**cycle_num

        # cosine decay with restarts
        min_lr = group["lr_min"]
        t = min(cycle_step / self.lr_cycle_steps, 1.0)  # clamp to 1.0 max
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(t * np.pi))
        return lr

    def get_lr(self):
        n = self.last_epoch

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`."
            )

        if n < 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        if n < self.lr_warm_up_steps:
            cycle_num, cycle_step = -1, n
        else:
            cycle_num, cycle_step = divmod(n - self.lr_warm_up_steps, self.lr_cycle_steps)

        if self.verbose and (n % self.verbose_interval == 0):
            logger.info(f"step={n}, cycle_step={cycle_step} last lr mult: {self.last_f}")

        return [self.get_group_lr(group, cycle_step, cycle_num) for group in self.optimizer.param_groups]


# TODO: process first_cycle_steps according to restarts set, skip cycle_mult, add min_lr to the args, ensure warmup
#  steps is properly set up to be handled according to restart, and allow gamma to be set, rename it to "decay" or
#  something
#  args to add to the UI: min_lr, gamma
class LegacyCosineAnnealingWarmupRestarts(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        min_lr: float | list[float] = 1e-6,
        warm_up_steps: int = 0,
        gamma: float = 0.9,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lrs: list[float] = []
        self.active_lrs: list[float] = []
        self.base_lrs: list[float] = []
        self.min_lrs = min_lr
        self.warm_up_steps = warm_up_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.step_in_cycle = last_epoch
        self.last_epoch = last_epoch
        self.cycle: int = 0
        self._last_lr = None

        if self.warm_up_steps >= self.first_cycle_steps:
            raise ValueError(
                f"[-] warm_up_steps must be smaller than first_cycle_steps. "
                f"{self.warm_up_steps} < {self.first_cycle_steps}"
            )

        super().__init__(optimizer, last_epoch, verbose)

        self.init_lr()

    def init_lr(self):
        self.max_lrs.clear()
        self.active_lrs.clear()
        self.base_lrs.clear()

        if not isinstance(self.min_lrs, list):
            self.min_lrs = [self.min_lrs] * len(self.optimizer.param_groups)

        for idx, param_group in enumerate(self.optimizer.param_groups):
            init_lr = param_group["initial_lr"]
            base_lr = self.min_lrs[idx] if init_lr > self.min_lrs[idx] else 0.0

            self.max_lrs.append(init_lr)
            self.active_lrs.append(init_lr)
            self.base_lrs.append(base_lr)
            param_group["lr"] = base_lr

    def get_lr(self) -> list[float]:
        if self.step_in_cycle == -1:
            return self.base_lrs

        if self.step_in_cycle < self.warm_up_steps:
            output = []
            for max_lr, base_lr in zip(self.active_lrs, self.base_lrs):
                output.append((max_lr - base_lr) * self.step_in_cycle / self.warm_up_steps + base_lr)
            return output

        output = []
        for max_lr, base_lr in zip(self.active_lrs, self.base_lrs):
            t = (self.step_in_cycle - self.warm_up_steps) / (self.cur_cycle_steps - self.warm_up_steps)
            output.append(base_lr + (max_lr - base_lr) * (1 + np.cos(t * np.pi)) / 2.0)
        return output

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warm_up_steps) * self.cycle_mult) + self.warm_up_steps
                )
        elif epoch >= self.first_cycle_steps:
            if self.cycle_mult == 1.0:
                self.step_in_cycle = epoch % self.first_cycle_steps
                self.cycle = epoch // self.first_cycle_steps
            else:
                n: int = int(
                    np.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult)
                )
                self.cycle = n
                self.step_in_cycle = epoch - int(
                    self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)
                )
                self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult**n
        else:
            self.cur_cycle_steps = self.first_cycle_steps
            self.step_in_cycle = epoch

        for i in range(len(self.active_lrs)):
            self.active_lrs[i] = self.max_lrs[i] * (self.gamma**self.cycle)

        self.last_epoch = np.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
