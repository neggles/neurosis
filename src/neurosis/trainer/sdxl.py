import logging
from os import isatty
from pathlib import Path
from typing import Annotated, List, NoReturn, Optional

import torch
import typer
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from lightning_utilities.core.imports import module_available
from rich.logging import RichHandler
from rich.pretty import install as install_pretty
from rich.traceback import install as install_traceback

from neurosis import __version__, console
from neurosis.models.diffusion import DiffusionEngine

train_app: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
)

logging.basicConfig(handlers=[RichHandler(console=console)], level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionTrainerCli(LightningCLI):
    pass


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
    if isatty(1):
        _, _ = install_pretty(console=console), install_traceback(console=console)

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
    )


if __name__ == "__main__":
    main()
