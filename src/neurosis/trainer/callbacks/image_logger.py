import logging
from enum import Enum
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
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast

from neurosis.models.utils import load_vae_ckpt
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
        ref_model_cls: Optional[str] = None,
        ref_model_ckpt: Optional[str] = None,
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

        # load ref model if provided
        self.ref_model = None
        if ref_model_ckpt is not None and ref_model_cls is not None:
            ref_model_ckpt = Path(ref_model_ckpt).resolve()
            if "vae" in ref_model_cls.lower():
                logger.info(f"loading reference model from {ref_model_ckpt}")
                self.ref_model = load_vae_ckpt(ref_model_ckpt, asymmetric="asym" in ref_model_cls)
            else:
                raise NotImplementedError(f"ref_model_cls {ref_model_cls} is not implemented yet")
        elif ref_model_cls is not None:
            raise ValueError("ref_model_cls provided but ref_model_ckpt is None")

        if self.ref_model is not None:
            self.ref_model = self.ref_model.eval().cpu().requires_grad_(False)

        self.__last_logged_step: int = -1

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
            case StepType.batch_idx:
                return batch_idx
            case StepType.global_step:
                return global_step
            case StepType.global_batch:
                return batch_idx * self.__trainer.accumulate_grad_batches
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
        if (step_idx % self.every_n_train_steps) == 0:
            return True

        # otherwise, don't log this step
        return False

    @rank_zero_only
    def log_local(
        self,
        split: str,
        images: dict[str, np.ndarray | Tensor] = {},
        batch: dict[str, np.ndarray | Tensor | str] = {},
        step: int = ...,
        epoch: int = ...,
        batch_idx: int = ...,
        pl_module: Optional[LightningModule] = None,
    ):
        save_dir = self.local_dir.joinpath("images", split)
        save_dir.mkdir(exist_ok=True, parents=True)

        wandb_dict = {"trainer/global_step": step}
        table_dict = {}

        for k in images:
            imgpath = save_dir / f"{k}_gs-{step:06}_e-{epoch:06}_b-{batch_idx:06}.png"
            if isinstance(images[k], Tensor) and images[k].ndim == 4 and images[k].shape[1] == 3:
                images[k] = [pt_to_pil(x) for x in images[k]]

            if k == "samples" and "caption" in batch:
                captions = batch["caption"]
                capgrid: CaptionGrid = CaptionGrid()
                img: Image.Image = capgrid(
                    images[k], captions, title=f"GS{step:06} E{epoch:06} B{batch_idx:06} samples"
                )
                img.save(imgpath)
                wandb_dict.update({f"{split}/{k}": wandb.Image(img, caption="Sample Grid")})

            elif isinstance(images[k][0], Image.Image):
                wandb_dict.update({f"{split}/{k}": [wandb.Image(x) for x in images[k]]})

            else:
                batch[k] = images[k]

        for k in [x for x in batch if x in self.extra_log_keys]:
            if isinstance(batch[k], list):
                if isinstance(batch[k][0], (str, np.bytes_)):
                    table_dict[k] = np_text_decode(batch[k], aslist=True)
                if isinstance(batch[k][0], Tensor):
                    # mildly hacky, turn tensor list back into tensor so the thing below can process it
                    if batch[k][0].ndim == 3 and batch[k][0].shape[0] == 3:
                        batch[k] = torch.stack(batch[k], dim=0)
                    # and if you passed in a list of batch-1 images, turn it into a batch
                    # you should not be doing this and i should not be allowing it but here we are
                    elif batch[k][0].ndim == 4 and batch[k][0].shape[1] == 3:
                        batch[k] = torch.cat(batch[k], dim=0)
                    elif batch[k][0].ndim == 2:
                        batch[k] = [tuple(x.cpu().tolist()) for x in batch[k]]
                    elif batch[k][0].ndim == 1:
                        batch[k] = [tuple(x.cpu().tolist()) for x in batch[k]]
                elif isinstance(batch[k][0], Image.Image):
                    for image in images[k]:
                        imgpath = save_dir / f"{k}_gs-{step:06}_e-{epoch:06}_b-{batch_idx:06}.png"
                        image.save(imgpath)
                    batch[k] = [wandb.Image(x) for x in images[k]]

            if isinstance(batch[k], Tensor):
                batch[k] = batch[k].detach().cpu()
                if (batch[k].ndim == 4 and batch[k].shape[1] == 3) or (
                    batch[k].ndim == 3 and batch[k].shape[0] == 3
                ):
                    batch[k] = pt_to_pil(batch[k], aslist=True)
                    batch[k] = [wandb.Image(x) for x in batch[k]]
                else:
                    del batch[k]

            if isinstance(batch[k], list):
                table_dict[k] = batch[k]
            else:
                warn(f"batch[{k}] is not a list, not logging this key to table_dict")

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

        # set up autocast kwargs
        autocast_kwargs = dict(
            device_type="cuda",
            enabled=self.enable_autocast,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        )
        # call the actual log_images method
        with torch.inference_mode(), autocast(**autocast_kwargs):
            images: dict[str, Tensor] = pl_module.log_images(
                batch, num_img=self.max_images, split=split, **self.log_func_kwargs
            )
            if self.ref_model is not None:
                logger.info(f"running reference model diff on {pl_module.__class__.__name__}")
                images = self.vae_reference_recons(images, num_img=self.max_images)

        # if the model returned None, warn and return early
        if images is None:
            warn(f"{pl_module.__class__.__name__} returned None from log_images")
            return

        for k in images:
            images[k] = images[k]
            if isinstance(images[k], Tensor):
                images[k] = images[k].detach().float().cpu()
                if self.clamp:
                    images[k] = images[k].clamp(min=-1.0, max=1.0)
                if self.rescale:
                    images[k] = (images[k] + 1.0) / 2.0

        # log the images
        self.log_local(
            split,
            images,
            batch,
            trainer.global_step,
            pl_module.current_epoch,
            batch_idx,
            pl_module=pl_module,
        )

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        if self.enabled:
            self.maybe_log_images(trainer, pl_module, batch, batch_idx, split="train")

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx, *args, **kwargs
    ):
        if self.enabled:
            self.maybe_log_images(trainer, pl_module, batch, batch_idx, split="val")

    @rank_zero_only
    def vae_reference_recons(
        self,
        images: dict[str, Tensor],
        num_img: int = 1,
        **kwargs,
    ) -> dict[str, Tensor]:
        if self.ref_model is None:
            return images

        inputs: Tensor = images.get("inputs")
        recons: Tensor = images.get("recons")

        with torch.inference_mode():
            recons = recons.detach().clone().to(self.ref_model.device, self.ref_model.dtype)
            inputs = inputs.detach().clone().to(self.ref_model.device, self.ref_model.dtype)

            ref_recons = [self.ref_model.forward(x.unsqueeze(0)).sample.squeeze(0) for x in inputs[:num_img]]
            if len(ref_recons) == 1:
                ref_recons = ref_recons[0].unsqueeze(0)
            else:
                ref_recons = torch.stack(ref_recons, dim=0)

            ref_diff = torch.clamp(ref_recons, -1.0, 1.0).sub(recons).abs().mul(0.5).clamp(0.0, 1.0)
            ref_diff_boost = ref_diff.mul(3.0).clamp(0.0, 1.0)
            ref_diff_input = torch.clamp(ref_recons, -1.0, 1.0).sub(inputs).abs().mul(0.5).clamp(0.0, 1.0)

        ref_dict = {
            "ref/recons": recons,
            "ref/diff_input": 2.0 * ref_diff_input - 1.0,
            "ref/diff": 2.0 * ref_diff - 1.0,
            "ref/diff_boost": 2.0 * ref_diff_boost - 1.0,
        }
        images.update(ref_dict)
        return images


def is_vae_model(model: nn.Module) -> bool:
    return all(
        (
            hasattr(model, "forward"),
            hasattr(model, "sample"),
            hasattr(model, "encode"),
            hasattr(model, "decode"),
        )
    )
