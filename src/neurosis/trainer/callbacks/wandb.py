import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from wandb.wandb_run import Run

logger = logging.getLogger(__name__)


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        if isinstance(trainer.logger, WandbLogger):
            config = self.parser.dump(self.config, format="yaml", skip_none=False)

            run: Run = trainer.logger.experiment
            with TemporaryDirectory(prefix="neurosis") as td:
                fp = Path(td).joinpath("config.yaml")
                fp.write_text(config)
                run.log_artifact(fp, type="config")

            config_dict = yaml.safe_load(config)
            try:
                trainer.logger.log_hyperparams(config_dict)
            except Exception:
                logger.exception("Failed to log hyperparameters to wandb as dict, logging as str")
                trainer.logger.log_hyperparams({"config": config})

            run.log_code(root=Path.cwd().joinpath("src"))

        return super().save_config(trainer, pl_module, stage)
