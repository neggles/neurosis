import logging
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np
import torch
import torchvision
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor
from torch.amp.autocast_mode import autocast

from neurosis.utils import isheatmap

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    global_step = "global_step"  # default
    batch_idx = "batch_idx"  # batch index instead of global step
    global_batch = "global_batch"  # global step * accumulate_grad_batches
    sample_idx = "sample_idx"  # global step * accumulate_grad_batches * batch_size


class ImageLogger(Callback):
    def __init__(
        self,
        every_n_train_steps: int = 100,
        # every_n_epochs: int = 1, # doesn't work without wasting memory
        max_images: int = 4,
        clamp: bool = True,
        rescale: bool = True,
        log_step_type: StepType = StepType.global_step,
        log_before_start: bool = False,
        log_first_step: bool = False,
        log_func_kwargs: dict = {},
        disabled: bool = False,
        enable_autocast: bool = True,
        batch_size: int = 1,
    ):
        super().__init__()
        self.every_n_train_steps = every_n_train_steps
        # self.every_n_epochs = every_n_epochs
        self.max_images = max_images
        self.rescale = rescale
        self.clamp = clamp
        self.enable_autocast = enable_autocast
        self.enabled = not disabled

        if self.max_images < 1 and self.enabled:
            raise ValueError("max_images must be >= 1 if disable=False")

        self.log_step_type = StepType(log_step_type)
        self.log_before_start = log_before_start
        self.log_first_step = log_first_step
        self.log_func_kwargs = log_func_kwargs
        self.batch_size = batch_size

        self.__last_logged_step: int = -1
        self.__trainer: Trainer = None
        self.__pl_module: LightningModule = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.__trainer = trainer
        self.__pl_module = pl_module

    @property
    def local_dir(self) -> Optional[Path]:
        tgt_dir = (  # pick the first one of these that's not None
            self.__pl_module.logger.save_dir
            or self.__pl_module.logger.log_dir
            or self.__trainer.log_dir
            or self.__trainer.default_root_dir
            or None
        )
        return Path(tgt_dir) if tgt_dir is not None else None

    def get_step_idx(self, batch_idx: int, global_step: Optional[int] = None) -> int:
        global_step = global_step or self.__trainer.global_step
        match self.log_step_type:
            case StepType.global_step:
                return global_step
            case StepType.batch_idx:
                return batch_idx
            case StepType.global_batch:
                return global_step * self.__trainer.accumulate_grad_batches
            case StepType.sample_idx:
                return batch_idx * self.__trainer.accumulate_grad_batches * self.batch_size
            case _:
                raise ValueError(f"invalid log_step_type: {self.log_step_type}")

    def check_step_idx(self, batch_idx: int, global_step: Optional[int] = None) -> bool:
        step_idx = self.get_step_idx(batch_idx, global_step)

        if step_idx == 0:
            return self.log_first_step

        # don't log the same step twice
        if step_idx == self.__last_logged_step:
            return False

        # check if step_idx is a multiple of every_n_train_steps
        if not (step_idx % self.every_n_train_steps):
            return True

        # otherwise, don't log this step
        return False

    @rank_zero_only
    def log_local(
        self,
        save_dir: PathLike,
        split: str,
        images: Union[Tensor, dict[str, Tensor]],
        global_step: int,
        current_epoch: int,
        batch_idx: int,
        pl_module: Optional[LightningModule] = None,
    ):
        if save_dir is None:
            return
        root = Path(save_dir).joinpath("images", split)
        root.mkdir(exist_ok=True, parents=True)

        for k in images:
            if isheatmap(images[k]):
                fig, ax = plt.subplots()
                ax = ax.matshow(images[k].cpu().numpy(), cmap="hot", interpolation="lanczos")
                plt.colorbar(ax)
                plt.axis("off")

                filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
                path = root / filename

                plt.savefig(path)
                plt.close()

                for logger in pl_module.loggers:
                    if isinstance(logger, WandbLogger):
                        img = Image.open(path)
                        logger.log_image(
                            key=f"{split}/{k}",
                            images=[img],
                            step=pl_module.global_step,
                        )
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid: np.ndarray = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)

                filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
                path = root / filename

                img = Image.fromarray(grid)
                img.save(path)

                for logger in pl_module.loggers:
                    if isinstance(logger, WandbLogger):
                        logger.log_image(
                            key=f"{split}/{k}",
                            images=[img],
                            step=pl_module.global_step,
                        )

    @rank_zero_only
    def maybe_log_images(
        self,
        pl_module: LightningModule,
        batch: Union[Tensor, dict[str, Tensor]],
        batch_idx: int,
        split: str = "train",
    ):
        # if max_images is 0 or we're disabled, do nothing
        if (not self.enabled) or (self.max_images == 0):
            return

        # check if we should log this step and return early if not
        if not self.check_step_idx(batch_idx, pl_module.global_step):
            return

        # now make sure the module has a log_images method that we can call
        if not hasattr(pl_module, "log_images"):
            warn(f"{pl_module.__class__.__name__} has no log_images method")
            return
        if not callable(pl_module.log_images):
            warn(f"{pl_module.__class__.__name__}'s log_images method is not callable! ")
            return

        # confirmed we're logging, save the step number
        self.__last_logged_step = self.get_step_idx(batch_idx, pl_module.global_step)

        # if the model is in training mode, flip to eval mode
        training = pl_module.training
        if training:
            pl_module.eval()

        # set up autocast kwargs
        autocast_kwargs = dict(
            device_type="cuda",
            enabled=self.enable_autocast,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        )
        # call the actual log_images method
        with torch.inference_mode(), autocast(**autocast_kwargs):
            images: list[Tensor] = pl_module.log_images(batch, split=split, **self.log_func_kwargs)

        # flip back to training mode if we flipped to eval mode earlier
        if training:
            pl_module.train()

        # if the model returned None, warn and return early
        if images is None:
            warn(f"{pl_module.__class__.__name__} returned None from log_images")
            return

        for k in images:
            # take the first max_images images from the sample batch, unless it's a heatmap
            if not isheatmap(images[k]):
                num_imgs = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:num_imgs]

            # detach, move to cpu, and clamp range if necessary
            if isinstance(images[k], Tensor):
                images[k] = images[k].detach().float().cpu()
                if self.clamp and not isheatmap(images[k]):
                    images[k] = torch.clamp(images[k], -1.0, 1.0)

            # log the images
            self.log_local(
                self.local_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module=pl_module,
            )

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        if self.enabled and pl_module.global_step > 0:
            self.maybe_log_images(pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx,
    ):
        if self.log_before_start and self.get_step_idx(batch_idx, pl_module.global_step) == 0:
            logger.info(f"{self.__class__.__name__}: logging before training")
            self.maybe_log_images(pl_module, batch, batch_idx, split="train", force=True)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        *args,
        **kwargs,
    ):
        if self.enabled and pl_module.global_step > 0:
            self.maybe_log_images(pl_module, batch, batch_idx, split="val")

        # # this isn't actually implemented here
        # calibrate = getattr(pl_module, "calibrate_grad_norm", False)
        # if calibrate and (batch_idx % 25 == 0) and batch_idx > 0:
        #     self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
