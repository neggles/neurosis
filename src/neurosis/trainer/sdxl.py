import logging
import os
from os import getenv, isatty
from pathlib import Path
from typing import Annotated, List, Optional

import jsonargparse
import torch
import typer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from rich.logging import RichHandler
from rich.pretty import install as install_pretty
from rich.traceback import install as install_traceback

from neurosis import __version__, console, is_debug
from neurosis.models.diffusion import DiffusionEngine
from neurosis.trainer.callbacks.image_logger import ImageLogger
from neurosis.trainer.callbacks.wandb import LoggerSaveConfigCallback

# set up rich if we're in a tty/interactive
if isatty(1):
    _ = install_pretty(console=console)
    _ = install_traceback(
        console=console,
        suppress=[jsonargparse, torch],
        show_locals=is_debug,
        locals_max_length=3,
        locals_max_string=64,
    )
del install_pretty, install_traceback

train_app: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich" if isatty(1) else None,
)

logging.basicConfig(
    handlers=[RichHandler(console=console)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DiffusionTrainerCli(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(
            lightning_class=ModelCheckpoint,
            nested_key="model_checkpoint",
        )
        parser.add_lightning_class_args(
            lightning_class=ImageLogger,
            nested_key="image_logger",
        )
        parser.add_lightning_class_args(
            lightning_class=LearningRateMonitor,
            nested_key="learning_rate_logger",
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

    if Path.cwd().joinpath("configs/lightning/defaults.yaml").exists():
        default_config_files: List[str] = ["configs/lightning/defaults.yaml"]
    else:
        default_config_files: List[str] = []

    torch.set_float32_matmul_precision("high")  # it whines if we don't do this

    cli = DiffusionTrainerCli(
        datamodule_class=None,
        model_class=DiffusionEngine,
        subclass_mode_data=True,
        subclass_mode_model=True,
        auto_configure_optimizers=False,
        args=args,
        parser_kwargs=dict(
            default_config_files=default_config_files,
        ),
        save_config_callback=LoggerSaveConfigCallback,
    )


if __name__ == "__main__":
    main()
