from os import PathLike, environ
from pathlib import Path
from typing import Optional

import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only

__run = None
__logger = None


@rank_zero_only
def init_wandb(
    save_dir: PathLike,
    config: Optional[dict] = None,
    project_name: Optional[str] = None,
    group_name: Optional[str] = None,
    run_name: Optional[str] = None,
    debug: bool = False,
):
    global __run
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"setting WANDB_DIR to {save_dir}")
    environ["WANDB_DIR"] = save_dir

    if debug:
        __run = wandb.init(project=project_name, mode="offline", group=group_name)
    else:
        __run = wandb.init(
            project=project_name,
            config=config,
            settings=wandb.Settings(code_dir="./sgm"),
            group=group_name,
            name=run_name,
        )
    print(f"wandb initialized for run {wandb.run.name}")


def get_wandb_logger(
    save_dir: PathLike,
    config: Optional[dict] = None,
    project_name: Optional[str] = None,
    group_name: Optional[str] = None,
    run_name: Optional[str] = None,
    debug: bool = False,
):
    global __logger
    if __logger is None:
        __logger = WandbLogger(
            name=run_name,
            save_dir=save_dir,
            project=project_name,
            group_name=group_name,
            config=config,
            offline=debug,
        )
    return __logger


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})
