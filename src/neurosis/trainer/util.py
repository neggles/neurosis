import logging
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Optional

from lightning.pytorch import Trainer
from natsort import natsorted
from packaging.version import Version

logger = logging.getLogger(__name__)


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
