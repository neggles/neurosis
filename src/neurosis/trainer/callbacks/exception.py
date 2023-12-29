import logging
from os import PathLike
from pathlib import Path
from typing import Optional

from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class ExceptionHandlerCallback(Callback):
    def __init__(
        self,
        default_dir: PathLike,
        default_name: str = "last",
        suffix: str = "exception",
    ):
        super().__init__()
        self.default_dir = Path(default_dir)
        self.default_name = default_name
        self.suffix = suffix if suffix.startswith(".") else "." + suffix

        # have to do this here in case of exception before setup()
        self.ckpt_dir: Optional[Path] = None
        self.ckpt_name: Optional[str] = None
        self.default_dir.mkdir(exist_ok=True, parents=True)

        # vars we'll set up in setup
        self.setup_done = False
        self.pl_module: Optional[LightningModule] = None
        self.trainer: Optional[Trainer] = None
        self.stage: Optional[str] = None

    @property
    def ckpt_path(self) -> Path:
        ckpt_dir = self.ckpt_dir or self.default_dir
        ckpt_name = f"{self.ckpt_name or self.default_name}_{self.suffix}"

        if self.trainer is not None:
            ckpt_name = f"{ckpt_name}.e{self.trainer.current_epoch:02d}s{self.trainer.global_step:08d}"

        return ckpt_dir.joinpath(ckpt_name + ".ckpt")

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.pl_module = pl_module
        self.trainer = trainer
        self.stage = stage

        self.ckpt_dir = Path(trainer.default_root_dir).joinpath("checkpoints")
        self.ckpt_name = trainer.checkpoint_callback.filename

        self.setup_done = True

    # called on exception, as the name implies (may be before setup())
    def on_exception(
        self, trainer: Trainer, pl_module: Optional[LightningModule], exception: BaseException
    ) -> None:
        if trainer.model is None:
            return

        logger.warning("Exception occurred after model load, saving checkpoint...")
        # overwrite if necessary
        trainer.save_checkpoint(self.ckpt_path)
        logger.warning("Checkpoint saved.")

        try:
            # gather some extra data if we can
            logger.warning("Dumping exception info to text file (with locals)...")
            from rich.console import Console

            exc_dumpfile = self.ckpt_path.with_suffix(".exception.txt")
            temp_console = Console(file=exc_dumpfile.open("w", encoding="utf-8"))
            temp_console.print_exception(width=130, extra_lines=5, show_locals=True)
        except Exception:
            logger.exception("Exception occurred while dumping exception info!")
            pass

    # Teardown is only called if we *don't* have an exception.
    def teardown(self, trainer: Trainer, *args, **kwargs) -> None:
        trainer.strategy.remove_checkpoint(self.ckpt_path)
