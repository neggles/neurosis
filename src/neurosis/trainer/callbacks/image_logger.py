import logging
from enum import Enum
from math import ceil, sqrt
from os import PathLike
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np
import torch
import wandb
from diffusers.image_processor import VaeImageProcessor
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from PIL import Image
from torch import Tensor
from torch.amp.autocast_mode import autocast
from torchvision.utils import make_grid

from neurosis.utils import isheatmap, ndimage_to_u8
from neurosis.utils.image.convert import pt_to_pil
from neurosis.utils.image.grid import CaptionGrid
from neurosis.utils.text import np_text_decode

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
        extra_log_keys: list[str] = [],
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
        self.extra_log_keys = extra_log_keys
        self.batch_size = batch_size

        self.__last_logged_step: int = -1
        self.__trainer: Trainer = None
        self.processor = VaeImageProcessor(do_resize=False, vae_scale_factor=8, do_convert_rgb=True)

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
                logger.debug("no save_dir, log_dir, or default_root_dir found...")

        return tgt_dir

    def get_step_idx(self, global_step: int, batch_idx: int) -> int:
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

    def check_step_idx(self, global_step: int, batch_idx: int) -> bool:
        step_idx = self.get_step_idx(global_step, batch_idx)

        # don't log the same step twice
        if step_idx <= self.__last_logged_step:
            return False

        # log the first step if log_first_step is True
        if step_idx == 0:
            return self.log_first_step

        # check if step_idx is a multiple of every_n_train_steps
        if step_idx > 0 and (step_idx % self.every_n_train_steps) == 0:
            return True

        # otherwise, don't log this step
        return False

    @rank_zero_only
    def log_local(
        self,
        save_dir: PathLike,
        split: str,
        log_dict: dict[str, Tensor],
        step: int,
        epoch: int,
        batch_idx: int,
        pl_module: Optional[LightningModule] = None,
    ):
        if save_dir is None:
            return
        save_dir = Path(save_dir).joinpath("images", split)
        save_dir.mkdir(exist_ok=True, parents=True)

        wandb_dict = {"trainer/global_step": step}
        if log_strings := log_dict.get("strings", None):
            if "caption" in log_strings and "samples" in log_dict:
                samples = log_dict.pop("samples")
                if isinstance(samples, Tensor):
                    samples = pt_to_pil(samples)

                log_strings["samples"] = [wandb.Image(x, mode="RGB") for x in samples]

                captions = log_strings["caption"]
                if isinstance(captions, list):
                    captions = [captions]

                grid = CaptionGrid()(
                    samples, captions, title=f"E{epoch:06} S{step:06} B{batch_idx:06} samples"
                )
                log_dict["samples"] = grid

            table = wandb.Table()
            for k in log_strings:
                table.add_column(k, log_strings[k])
            wandb_dict[f"{split}/sample_table"] = table

        for k, val in log_dict.items():
            imgpath = save_dir / f"{k}_gs-{step:06}_e-{epoch:06}_b-{batch_idx:06}.png"
            img = None
            if isinstance(val, Image.Image):
                img = val
            elif isheatmap(val):
                fig, ax = plt.subplots()
                ax.set_axis_off()
                img: AxesImage = ax.matshow(val.cpu().numpy(), cmap="hot", interpolation="lanczos")
                fig.colorbar(img, ax=ax)
                fig.savefig(imgpath)
                plt.close(fig)
                img = Image.open(imgpath)

            elif isinstance(val, Tensor):
                if val.ndim == 3:
                    val = val.unsqueeze(0)
                try:
                    img = ndimage_to_u8(make_grid(val, nrow=ceil(sqrt(val.shape[0]))).cpu().numpy())
                    img = Image.fromarray(img)
                    img.save(imgpath)
                except Exception as e:
                    logger.exception(e)
                    continue

            if img is not None:
                wandb_dict.update({f"{split}/{k}": wandb.Image(img)})

        if pl_module is not None:
            for pl_logger in [x for x in pl_module.loggers if isinstance(x, WandbLogger)]:
                pl_logger.log_metrics(metrics=wandb_dict)

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
        if (not self.enabled) or (self.max_images == 0):
            return

        # check if we should log this step and return early if not
        if not self.check_step_idx(trainer.global_step, batch_idx):
            return

        # now make sure the module has a log_images method that we can call
        if not hasattr(pl_module, "log_images"):
            warn(f"{pl_module.__class__.__name__} has no log_images method")
            return
        if not callable(pl_module.log_images):
            warn(f"{pl_module.__class__.__name__}'s log_images method is not callable! ")
            return

        # confirmed we're logging, save the step number
        self.__last_logged_step = self.get_step_idx(trainer.global_step, batch_idx)

        # set up autocast kwargs
        autocast_kwargs = dict(
            device_type="cuda",
            enabled=self.enable_autocast,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        )
        # call the actual log_images method
        with torch.inference_mode(), autocast(**autocast_kwargs):
            log_dict: list[Tensor] = pl_module.log_images(
                batch, num_img=self.max_images, split=split, **self.log_func_kwargs
            )

        # if the model returned None, warn and return early
        if log_dict is None:
            warn(f"{pl_module.__class__.__name__} returned None from log_images")
            return

        for k in log_dict:
            if isinstance(log_dict[k], Tensor):
                log_dict[k] = log_dict[k].detach().float().cpu()
            if not isheatmap(log_dict[k]):
                log_dict[k] = log_dict[k][: min(log_dict[k].shape[0], self.max_images)]
                if self.clamp:
                    log_dict[k] = log_dict[k].clamp(min=-1.0, max=1.0)
                if self.rescale:
                    log_dict[k] = (log_dict[k] + 1.0) / 2.0

        log_strings = {}
        for k in self.extra_log_keys:
            try:
                if k in batch:
                    log_strings[k] = batch[k][: self.max_images]
                    if isinstance(log_strings[k][0], np.bytes_):
                        log_strings[k] = np_text_decode(log_strings[k], aslist=True)

                    log_strings[k] = [np_text_decode(x) for x in log_strings[k]]
            except Exception as e:
                logger.exception(e)
                continue

        if len(log_strings) > 0:
            log_dict["strings"] = log_strings

        # log the images
        self.log_local(
            self.local_dir,
            split,
            log_dict,
            trainer.global_step,
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
        if self.enabled:
            self.maybe_log_images(trainer, pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch,
        batch_idx,
    ):
        if self.enabled and self.log_before_start and trainer.global_step == 0:
            logger.info(f"{self.__class__.__name__} running log before training...")
            self.maybe_log_images(trainer, pl_module, batch, batch_idx, split="train", force=True)

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
        if self.enabled and trainer.global_step > 0:
            self.maybe_log_images(trainer, pl_module, batch, batch_idx, split="val")
