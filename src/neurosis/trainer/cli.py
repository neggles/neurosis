import logging
from os import getenv, isatty
from pathlib import Path
from typing import Annotated, Optional
from warnings import filterwarnings

import jsonargparse
import lightning
import torch
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningArgumentParser, LightningCLI
from rich.traceback import install as install_traceback

from neurosis import __version__, console, is_debug
from neurosis.trainer.callbacks.image_logger import ImageLogger
from neurosis.trainer.callbacks.wandb import LoggerSaveConfigCallback

# set up rich if we're in a tty/interactive and NOT in kube
if isatty(1) and getenv("KUBERNETES_PORT", None) is not None:
    _ = install_traceback(
        console=console,
        suppress=[jsonargparse, torch, lightning],
        show_locals=is_debug,
        locals_max_length=1,
        locals_max_string=64,
        max_frames=20,
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

    def before_instantiate_classes(self) -> None:
        if self.config.subcommand != "fit":
            logger.debug(f"Skipping before_instantiate_classes hook for {self.config.subcommand}")
            return
        logger.debug("Running before_instantiate_classes hook")
        if trainer_args := self.config.get("fit", {}).get("trainer"):
            if hasattr(trainer_args, "default_root_dir"):
                default_root_dir = Path(trainer_args.default_root_dir).resolve()
                logger.info(f"Making sure trainer workdir {default_root_dir} exists")
                default_root_dir.mkdir(exist_ok=True, parents=True)

            if train_loggers := trainer_args.get("logger"):
                logger.info(f"Got loggers: {train_loggers}")
                for lobj in [
                    x
                    for x in train_loggers
                    if (
                        hasattr(x, "class_path")
                        and hasattr(x, "init_args")
                        and "wandb" in x.class_path.lower()
                    )
                ]:
                    if save_dir := lobj.init_args.get("save_dir"):
                        save_dir = Path(save_dir).resolve()
                        logger.info(f"Making sure wandb save_dir {save_dir} exists")
                        save_dir.mkdir(exist_ok=True, parents=True)


@train_app.command(add_help_option=False)
def main(
    args: Annotated[
        Optional[ArgsType],
        typer.Argument(help="Arguments to pass to the trainer."),
    ] = None,
):
    """
    Main entrypoint for training models.
    """
    logger.info(f"neurosis v{__version__} training script running!")
    logger.debug(f"PyTorch {torch.__version__}, Lightning {lightning.__version__}")

    # temporary since Diffusers triggers this warning on Torch 2.2.1 atm
    logger.debug("Filtering out torch.utils._pytree._register_pytree_node warning (caused by Diffusers)")
    filterwarnings("ignore", category=UserWarning, message=r"torch\.utils\._pytree\._register_pytree_node.*")

    allow_tf32 = getenv("NEUROSIS_DISABLE_TF32", "").lower() not in ["1", "true", "yes"]
    if allow_tf32:
        logger.info("Enabling matmul performance optimizations (tf32 and 'high' precision)")
        torch.set_float32_matmul_precision("high")
    else:
        logger.warning("Disabling matmul optimizations as NEUROSIS_DISABLE_TF32 is set")
        torch.set_float32_matmul_precision("highest")

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        logger.info(f"{'Enabled' if allow_tf32 else 'Disabled'} tf32 in CUDA")
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = allow_tf32
        logger.info(f"{'Enabled' if allow_tf32 else 'Disabled'} tf32 in cuDNN")

    cli = DiffusionTrainerCli(  # type: ignore
        datamodule_class=None,
        subclass_mode_data=True,
        subclass_mode_model=True,
        auto_configure_optimizers=False,
        args=args,
        trainer_defaults=dict(
            enable_model_summary=False,
        ),
        parser_kwargs=dict(
            default_env=True,
            parser_mode="omegaconf",
        ),
        save_config_callback=LoggerSaveConfigCallback,
        save_config_kwargs=dict(
            overwrite=True,
        ),
    )


if __name__ == "__main__":
    main()
