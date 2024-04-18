import logging
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Optional, Type

import lightning.pytorch as L
import torch
import torch.distributed as dist
from lightning.fabric.utilities.rank_zero import _get_rank, rank_zero_only
from lightning.pytorch import Trainer
from natsort import natsorted
from packaging.version import Version

logger = logging.getLogger(__name__)


class LocalRankZeroFilter(logging.Filter):
    """
    Filter that only allows rank 0 to log messages at or below a certain level.
    Used to ignore debug and info messages from non-zero ranks.
    """

    def __init__(self, min_level: int = logging.WARNING):
        self.min_level = min_level

    def filter(self, record):
        rank = getattr(rank_zero_only, "rank", _get_rank() or 0)
        if rank != 0 and record.levelno < self.min_level:
            return False
        return True


class AddRankFilter(logging.Filter):
    def __init__(self, rank: int = -1):
        self._rank = rank

    @property
    def rank(self) -> int:
        if self._rank == -1 and dist.is_initialized():
            self._rank = getattr(rank_zero_only, "rank", _get_rank() or -1)
        return self._rank

    def filter(self, record: logging.LogRecord) -> bool:
        if self.rank != -1:
            record.msg = f"[R{self.rank}] {record.msg}"
        return True


def instantiate_compile_class(class_type: Type[L.LightningModule], *args, **kwargs) -> L.LightningModule:
    compile_kwargs = kwargs.pop("compile_kwargs", None)

    module = class_type(*args, **kwargs)
    if compile_kwargs is not None:
        logger.info(f"Compiling module {module} with TorchDynamo...")
        module = torch.compile(module, **compile_kwargs)
    return module


class EMATracker:
    def __init__(
        self,
        alpha: Optional[float] = None,
        steps: Optional[int] = None,
    ):
        super().__init__()
        self._value = None
        if alpha is None and steps is None:
            raise ValueError("Either alpha or steps must be provided")

        if alpha is None:
            self.alpha = 2 / (steps + 1)
        else:
            self.alpha = alpha

    def __call__(self, new_value: float):
        self.update(new_value)

    def update(self, new_value: float):
        if self._value is None:
            self._value = new_value
            return
        # calculate new value
        self._value = new_value * self.alpha + self._value * (1 - self.alpha)

    @property
    def value(self):
        return self._value


def default_trainer_args() -> dict[str, Any]:
    argspec = dict(signature(Trainer.__init__).parameters)
    argspec.pop("self")
    default_args = {param: argspec[param].default for param in argspec if argspec[param] != Parameter.empty}
    return default_args


def get_checkpoint_name(logdir: Path) -> tuple[Path, str]:
    ckpt_dir = logdir.joinpath("checkpoints").resolve()
    ckpt_files = natsorted(ckpt_dir.glob("last**.ckpt"))

    logger.info('available "last" checkpoints:')
    logger.info([f"  - {x.absolute().relative_to(ckpt_dir)}" for x in ckpt_files])
    if len(ckpt_files) > 1:
        logger.info("got most recent checkpoint")
        last_ckpt = sorted(ckpt_files, key=lambda x: x.stat().st_mtime)[-1]
        logger.info(f"Most recent ckpt is {last_ckpt}")
        logdir.joinpath("most_recent_ckpt.txt").write_text(last_ckpt + "\n")
        try:
            version = Version(last_ckpt.stem.split("-v")).major
        except Exception as e:
            logger.exception("version confusion but not bad")
            version = 1
        # version = last_version + 1
    else:
        # in this case, we only have one "last.ckpt"
        ckpt_files = ckpt_files[0]
        version = 1
    melk_ckpt_name = f"last-v{version}.ckpt"
    logger.info(f"Current melk ckpt_files name: {melk_ckpt_name}")
    return ckpt_files, melk_ckpt_name
