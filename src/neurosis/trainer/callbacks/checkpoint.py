from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint


@lru_cache(maxsize=4)
def get_checkpoint_logger(ckpt_dir: PathLike, monitor: Optional[str] = None) -> ModelCheckpoint:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    save_top_k = 3 if monitor is not None else 1

    return ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:06d}-{step:06d}",
        verbose=True,
        save_last=True,
        save_on_train_epoch_end=True,
        monitor=monitor,
        save_top_k=save_top_k,
    )
