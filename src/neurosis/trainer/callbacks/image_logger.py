import logging
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np
import torch
import wandb
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch import Tensor
from torch.amp.autocast_mode import autocast

from neurosis.utils.image import CaptionGrid, label_batch, numpy_to_pil, pt_to_pil
from neurosis.utils.text import np_text_decode

from .common import BatchDictType, LogDictType, StepType

logger = logging.getLogger(__name__)


class ImageLogger(Callback):
    def __init__(
        self,
        every_n_train_steps: int = 100,
        max_images: int = 4,
        clamp: bool = True,
        rescale: bool = True,
        log_step_type: StepType = StepType.global_step,
        log_before_start: bool = False,
        log_first_step: bool = False,
        log_func_kwargs: dict = {},
        extra_log_keys: list[str] = [],
        disabled: bool = False,
        enable_autocast: bool = True,
        batch_size: int = 1,
        accumulate_grad_batches: int = 1,
        label_img: bool = False,
    ):
        super().__init__()
        self.every_n_train_steps = every_n_train_steps
        # self.every_n_epochs = every_n_epochs
        self.max_images = max_images
        self.rescale = rescale
        self.clamp = clamp
        self.enable_autocast = enable_autocast
        self.enabled = not disabled
        self.label_img = label_img

        if self.max_images < 1 and self.enabled:
            raise ValueError("max_images must be >= 1 if disable=False")

        self.log_step_type = StepType(log_step_type)
        self.log_before_start = log_before_start
        self.log_first_step = log_first_step
        self.log_func_kwargs = log_func_kwargs
        self.extra_log_keys = extra_log_keys
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches

        self.diff_boost = 3.0
        self.__last_logged_step: int = -1
        self.__trainer: Trainer = None

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.__trainer = trainer

    @property
    def local_dir(self) -> Optional[Path]:
        tgt_dir = None
        for pl_logger in self.__trainer.loggers:
            if pl_logger.save_dir is not None:
                logger.debug(f"using {pl_logger.__class__.__name__} save_dir")
                tgt_dir = Path(pl_logger.save_dir).resolve()
                break
            if pl_logger.log_dir is not None:
                logger.debug(f"using {pl_logger.__class__.__name__} log_dir")
                tgt_dir = Path(pl_logger.log_dir).resolve()
                break

        if tgt_dir is None:
            if self.__trainer.default_root_dir is not None:
                logger.debug(f"using {self.__trainer.__class__.__name__} default_root_dir")
                tgt_dir = Path(self.__trainer.default_root_dir)
            else:
                raise ValueError("no save_dir, log_dir, or default_root_dir found!")

        return tgt_dir

    def get_step_idx(self, global_step: int, batch_idx: int) -> int:
        match self.log_step_type:
            case StepType.global_step:
                return self.__trainer.global_step
            case StepType.batch_idx:
                return batch_idx
            case StepType.global_batch:
                return batch_idx * self.accumulate_grad_batches
            case StepType.sample_idx:
                return batch_idx * self.accumulate_grad_batches * self.batch_size
            case _:
                raise ValueError(f"invalid log_step_type: {self.log_step_type}")

    def check_step_idx(self, global_step: int, batch_idx: int) -> bool:
        step_idx = self.get_step_idx(global_step, batch_idx)
        # don't log the same step twice
        if step_idx <= self.__last_logged_step:
            return False
        # log the first step if log_first_step is True
        if step_idx == 0:
            return self.log_first_step
        # check if step_idx is a multiple of every_n_train_steps
        if (step_idx % self.every_n_train_steps) == 0:
            return True
        # otherwise, don't log this step
        return False

    def make_caption_grid(
        self,
        images: list[Image.Image],
        captions: list[str],
        title: Optional[str] = None,
        ncols: Optional[int] = None,
    ) -> Image.Image:
        if len(images) != len(captions):
            raise ValueError("Number of images and captions must match!")

        capgrid: CaptionGrid = CaptionGrid()
        return capgrid(images, captions, title, ncols)

    def rescale_pixel_values(self, images: LogDictType) -> LogDictType:
        for k in images:
            images[k] = images[k]
            if isinstance(images[k], Tensor):
                images[k] = images[k].detach().float().cpu()
                if self.clamp:
                    images[k] = images[k].clamp(min=-1.0, max=1.0)
                if self.rescale:
                    images[k] = images[k].add(1.0).div(2.0)
        return images

    @rank_zero_only
    def call_log_images(
        self,
        pl_module: LightningModule,
        batch: BatchDictType,
        batch_idx: int = -1,
        split: str = "train",
        num_img: int = 4,
    ) -> LogDictType:
        # set up autocast kwargs
        autocast_kwargs = dict(
            device_type="cuda",
            enabled=self.enable_autocast,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        )
        # call the actual log_images method
        with autocast(**autocast_kwargs):
            images: dict[str, Tensor] = pl_module.log_images(
                batch, num_img=num_img, split=split, **self.log_func_kwargs
            )

        return images

    @rank_zero_only
    def log_local(
        self,
        split: str,
        images: LogDictType = {},
        batch: BatchDictType = {},
        step: int = ...,
        epoch: int = ...,
        batch_idx: int = ...,
        pl_module: Optional[LightningModule] = None,
    ):
        save_dir = self.local_dir.joinpath("images", split)
        save_dir.mkdir(exist_ok=True, parents=True)

        fstem = f"gs{step:06d}_e{epoch:06d}_b{batch_idx:06d}"
        title = fstem.replace("-", "").replace("_", " ").upper()

        # apply range scaling and clamping to images
        images = self.rescale_pixel_values(images)

        wandb_dict = {"trainer/global_step": step}
        table_dict = {}

        def add_to_both(k: str, v):
            wandb_dict[k] = v
            table_dict[k] = v

        if "samples" in images:
            samples = pt_to_pil(images.pop("samples"), aslist=True)
            if self.label_img:
                samples = label_batch(samples, step, copy=True)

            for idx, img in enumerate(samples):
                img.save(save_dir / f"{fstem}_samples_{idx:02d}.png")

            if "caption" in batch:
                wandb_samples = [wandb.Image(img, caption=cap) for img, cap in zip(samples, batch["caption"])]
                try:
                    grid = self.make_caption_grid(samples, batch["caption"], title=title + " samples")
                    grid.save(save_dir.joinpath(f"{fstem}_samples_grid.png"))
                    wandb_dict[f"{split}/sample_grid"] = wandb.Image(grid, caption="Sample Grid")
                except Exception as e:
                    logger.exception("Failed to make sample grid, continuing", e)
            else:
                wandb_samples = [wandb.Image(img) for img in samples]
            add_to_both(f"{split}/samples", wandb_samples)

        for k in images:
            try:
                val = images[k]
                # if we have a single number, log it directly
                if isinstance(val, (int, float)):
                    wandb_dict[f"{split}/{k}"] = val
                    continue

                # cast images to PIL if they're not already
                if isinstance(val[0], Tensor):
                    if val[0].ndim == 3:
                        val = pt_to_pil(val, aslist=True)
                if isinstance(val[0], np.ndarray):
                    val = numpy_to_pil(val, aslist=True)

                # we should now have a list of PIL images, so save them
                if isinstance(val[0], Image.Image):
                    if self.label_img:
                        val = label_batch(val, step, copy=True)
                    for idx, img in enumerate(val):
                        img.save(save_dir / f"{fstem}_{k.replace('/', '_')}_{idx:02d}.png")
                    add_to_both(f"{split}/{k}", [wandb.Image(x) for x in images[k]])

            except Exception:
                logger.exception(f"Failed to log {k}, continuing")
                continue

        for k in [x for x in batch if x in self.extra_log_keys]:
            try:
                val = batch[k]
                if isinstance(val[0], Tensor):
                    if val[0].ndim == 3 and val[0].shape[0] == 3:
                        val = pt_to_pil(val, aslist=True)
                    elif val[0].ndim in [1, 2]:
                        val = [tuple(x.cpu().tolist()) for x in val]

                if isinstance(val[0], Image.Image):
                    if self.label_img:
                        val = label_batch(val, step, copy=True)
                    for idx, img in enumerate(val):
                        img.save(save_dir / f"{fstem}_{k.replace('/', '_')}_{idx:02d}.png")
                    val = [wandb.Image(x) for x in val]

                if isinstance(val, list):
                    table_dict[k] = val
                else:
                    warn(f"batch[{k}] is not a list, not logging this key to table_dict")

            except Exception:
                logger.exception(f"Failed to log {k}, continuing")
                continue

        if len(table_dict) > 0:
            table = wandb.Table(columns=[str(x) for x in table_dict.keys()], allow_mixed_types=True)
            for data in zip(*table_dict.values()):
                table.add_data(*data)
            wandb_dict.update({f"{split}/table": table})

        if pl_module is not None:
            for pl_logger in [x for x in pl_module.loggers if isinstance(x, WandbLogger)]:
                pl_logger.log_metrics(wandb_dict)

    @rank_zero_only
    def maybe_log_images(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Union[Tensor, dict[str, Tensor]],
        batch_idx: int,
        split: str = "train",
    ):
        # if max_images is 0 or we're disabled, do nothing
        if self.enabled is False or self.max_images < 1:
            return

        # check if we should log this step and return early if not
        if self.check_step_idx(trainer.global_step, batch_idx) is False:
            return

        # now make sure the module has a log_images method that we can call
        if not hasattr(pl_module, "log_images"):
            logger.warning(f"{pl_module.__class__.__name__} has no log_images method")
            return
        if not callable(pl_module.log_images):
            logger.warning(f"{pl_module.__class__.__name__}'s log_images method is not callable! ")
            return

        # confirmed we're logging, save the step number
        if split == "train":
            self.__last_logged_step = self.get_step_idx(trainer.global_step, batch_idx)

        # trim the batch to max_images and decode any text
        for k in batch:
            batch[k] = batch[k][: self.max_images]
            if isinstance(batch[k][0], (str, np.bytes_)):
                batch[k] = [np_text_decode(x) for x in batch[k]]

        # do the actual log image generation
        images = self.call_log_images(pl_module, batch, batch_idx, split=split, num_img=self.max_images)
        # if the model returned None, warn and return early
        if images is None:
            warn(f"{pl_module.__class__.__name__} returned None from log_images")
            return

        # log the images
        self.log_local(
            split, images, batch, trainer.global_step, pl_module.current_epoch, batch_idx, pl_module=pl_module
        )

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        self.maybe_log_images(trainer, pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, *args, **kwargs
    ):
        self.maybe_log_images(trainer, pl_module, batch, batch_idx, split="val")
