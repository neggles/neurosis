import logging
import time
from datetime import timedelta
from os import PathLike
from pathlib import Path
from typing import Any, Optional
from weakref import proxy

from huggingface_hub import Repository
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT

logger = logging.getLogger(__name__)


class HFHubCheckpoint(Checkpoint):
    def __init__(
        self,
        repo_id: str,
        token: Optional[bool | str] = None,
        ckpt_name: str = "model",
        save_last: bool = False,
        every_n_train_steps: Optional[int] = None,
        every_n_epochs: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        convert_to_diffusers: bool = False,
        verbose: bool = False,
    ):
        self.verbose = verbose

        self.repo_id = repo_id
        self.token = token
        self.ckpt_name = ckpt_name

        self.save_last = save_last
        self._save_on_train_epoch_end = save_on_train_epoch_end
        self._last_global_step_saved = 0  # no need to save when no steps were taken
        self._last_time_checked: Optional[float] = None
        self.last_model_path = ""

        self.convert_to_diffusers = convert_to_diffusers
        if self.convert_to_diffusers is True:
            raise NotImplementedError("Diffusers conversion is not yet implemented.")

        self._hf_repo: Repository = None
        self._temp_dir: Optional[PathLike] = None

        self.__init_triggers(every_n_train_steps, every_n_epochs, train_time_interval)
        self.__validate_init_configuration()

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            every_n_train_steps=self._every_n_train_steps,
            every_n_epochs=self._every_n_epochs,
            train_time_interval=self._train_time_interval,
        )

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self._temp_dir is None:
            self._temp_dir = Path(trainer.default_root_dir).joinpath(".hub")
            self._temp_dir.mkdir(exist_ok=True, parents=True)
        if self._hf_repo is None:
            self._hf_repo = Repository(
                local_dir=self._temp_dir,
                clone_from=self.repo_id,
                token=self.token,
                skip_lfs_files=True,
                revision=None,
            )
        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self._temp_dir)

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._last_time_checked = time.monotonic()

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        if self._should_skip_saving_checkpoint(trainer):
            return
        skip_batch = self._every_n_train_steps < 1 or (trainer.global_step % self._every_n_train_steps != 0)

        interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if interval:
            prev_time_check = self._last_time_checked
            skip_time = prev_time_check is None or (now - prev_time_check) < interval.total_seconds()
            # broadcast to other processes just in case of clock mismatches
            skip_time = trainer.strategy.broadcast(skip_time)

        if not skip_time:
            self._last_time_checked = now
            self._save_checkpoint(trainer, "time")

        if not skip_batch:
            self._save_checkpoint(trainer, "step")

        if trainer.is_last_batch and self.save_last:
            self._save_checkpoint(trainer, "last")

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                self._save_checkpoint(trainer, "epoch")
            self._save_checkpoint(trainer, "last")

    def _save_checkpoint(self, trainer: Trainer, event: str) -> None:
        if self._last_global_step_saved == trainer.global_step:
            logger.info(f"Checkpoint at step {trainer.global_step} already saved, skipping.")
            return

        if event in ["epoch", "step", "time"]:
            epoch_and_step = f"e{trainer.current_epoch}s{trainer.global_step}"
            ckpt_path = self._temp_dir.joinpath(f"{self.ckpt_name}_{epoch_and_step}.ckpt")
            commit_message = f"add/update {self.ckpt_name} at {epoch_and_step}"
        if event == "last":
            ckpt_path = self._temp_dir.joinpath(f"{self.ckpt_name}_{epoch_and_step}_last.ckpt")
            commit_message = f"add/update {self.ckpt_name} at end of run"
        else:
            raise ValueError(f"Unknown checkpoint event {event}")

        with self._hf_repo.commit(commit_message=commit_message, blocking=False, auto_lfs_prune=True):
            trainer.save_checkpoint(ckpt_path, weights_only=True)

        self._last_global_step_saved = trainer.global_step
        # notify loggers
        if trainer.is_global_zero:
            for pl_logger in trainer.loggers:
                pl_logger.after_save_checkpoint(proxy(self))

    def _should_skip_saving_checkpoint(self, trainer: Trainer) -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return (
            bool(trainer.fast_dev_run)  # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING  # don't save anything during non-fit
            or trainer.sanity_checking  # don't save anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already saved at the last step
        )

    def _should_save_on_train_epoch_end(self, trainer: Trainer) -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end
        # don't auto-save on every epoch if we're not running validation every epoch
        if trainer.check_val_every_n_epoch is not None and trainer.check_val_every_n_epoch != 1:
            return False
        # do auto-save on every epoch if there's no val loop at all
        if sum(trainer.num_val_batches) == 0:
            return True
        # otherwise if we're validating more than once per epoch, run after validation instead
        return trainer.val_check_interval == 1.0

    def __validate_init_configuration(self) -> None:
        if self._every_n_train_steps < 0:
            raise MisconfigurationException(
                f"Invalid value for every_n_train_steps={self._every_n_train_steps}. Must be >= 0"
            )
        if self.every_n_epochs < 0:
            raise MisconfigurationException(
                f"Invalid value for every_n_epochs={self.every_n_epochs}. Must be >= 0"
            )

        if self._every_n_train_steps >= 1 and self._train_time_interval is not None:
            raise MisconfigurationException(
                f"Combination of every_n_train_steps={self._every_n_train_steps} and "
                + f"train_time_interval={self._train_time_interval} is redundant and not supported."
            )

    def __init_triggers(
        self,
        every_n_train_steps: Optional[int],
        every_n_epochs: Optional[int],
        train_time_interval: Optional[timedelta],
    ) -> None:
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs = 1
            every_n_train_steps = 0
            logger.warning("Checkpoint interval not specified, defaulting to once per validation epoch")
        else:
            every_n_epochs = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0

        self._train_time_interval: Optional[timedelta] = train_time_interval
        self._every_n_epochs: int = every_n_epochs
        self._every_n_train_steps: int = every_n_train_steps

    @property
    def every_n_epochs(self) -> Optional[int]:
        return self._every_n_epochs

    def __warn_if_dir_not_empty(self, path: PathLike) -> None:
        path = Path(path)
        if any((x for x in path.iterdir() if not x.name.startswith("."))):
            logger.warning(f"Temporary directory {path} is not empty, any files in it may be pushed to HF!")
