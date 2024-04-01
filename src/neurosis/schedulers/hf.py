from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import optimization as O


class CosineWithHardRestartsAndWarmUp(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self._lr_lambda = partial(
            O._get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles,
        )
        super().__init__(optimizer, self._lr_lambda, last_epoch)


class CosineWithWarmUp(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self._lr_lambda = partial(
            O._get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles,
        )
        super().__init__(optimizer, self._lr_lambda, last_epoch)
