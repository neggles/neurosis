from functools import wraps

from lightning.pytorch.callbacks import LearningRateMonitor


@wraps(LearningRateMonitor)
def get_lr_monitor(**kwargs) -> LearningRateMonitor:
    kwargs.setdefault("logging_interval", "step")
    return LearningRateMonitor(**kwargs)
