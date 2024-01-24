import logging
from os import getenv, isatty
from pathlib import Path
from typing import Annotated, List, Optional

import jsonargparse
import lightning
import torch
import typer
from lightning.pytorch.callbacks import (  # noqa: F401
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger  # noqa: F401
from lightning.pytorch.loggers.wandb import WandbLogger  # noqa: F401
from rich.traceback import install as install_traceback

from neurosis import __version__, console, is_debug
from neurosis.models.diffusion import DiffusionEngine  # noqa: F401
from neurosis.trainer.callbacks.exception import ExceptionHandlerCallback  # noqa: F401
from neurosis.trainer.callbacks.image_logger import ImageLogger
from neurosis.trainer.callbacks.wandb import LoggerSaveConfigCallback

# set up rich if we're in a tty/interactive and NOT in kube
if isatty(1) and getenv("KUBERNETES_PORT", None) is not None:
    _ = install_traceback(
        console=console,
        suppress=[jsonargparse, torch, lightning],
        show_locals=is_debug,
        locals_max_length=3,
        locals_max_string=64,
    )
del install_traceback

train_app: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich" if isatty(1) else None,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionTrainerCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(
            lightning_class=ModelCheckpoint,
            nested_key="model_checkpoint",
            required=False,
        )
        parser.add_lightning_class_args(
            lightning_class=ImageLogger,
            nested_key="image_logger",
            required=False,
        )


@train_app.command(add_help_option=False)
def main(
    args: Annotated[
        Optional[ArgsType],
        typer.Argument(help="Arguments to pass to the trainer."),
    ] = None,
):
    """
    Main entrypoint for training Stable Diffusion models.
    """
    logger.info(f"neurosis v{__version__} training script running!")
    logger.debug(f"PyTorch {torch.__version__}, Lightning {lightning.__version__}")

    default_config_files: List[str] = []
    if Path.cwd().joinpath("configs/lightning/defaults.yaml").exists():
        default_config_files.append("configs/lightning/defaults.yaml")

    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        logger.info("Enabling TensorFloat32 in cuDNN library")
        torch.backends.cudnn.allow_tf32 = True

    torch.set_float32_matmul_precision("high")  # it whines if we don't do this

    cli = DiffusionTrainerCli(  # type: ignore
        datamodule_class=None,
        subclass_mode_data=True,
        subclass_mode_model=True,
        auto_configure_optimizers=False,
        args=args,
        parser_kwargs=dict(
            default_env=True,
            default_config_files=default_config_files,
            parser_mode="omegaconf",
        ),
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs=dict(
            overwrite=True,
        ),
    )


if __name__ == "__main__":
    main()
