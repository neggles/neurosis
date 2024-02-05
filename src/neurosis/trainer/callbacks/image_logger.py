import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import numpy as np
import torch
import wandb
from diffusers import AutoencoderKL
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch import Tensor, nn
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F

from neurosis.models.utils import load_vae_ckpt
from neurosis.modules.autoencoding.asymmetric import AsymmetricAutoencoderKL
from neurosis.utils.image.convert import numpy_to_pil, pt_to_pil
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
        accumulate_grad_batches: int = 1,
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
        self.accumulate_grad_batches = accumulate_grad_batches

        # load ref model if provided
        self.ref_model: nn.Module = None
        self.ref_model_ckpt: Optional[Path] = None
        self.ref_model_cls: Optional[str] = ref_model_cls
        if self.ref_model_cls is not None:
            if ref_model_ckpt is None:
                raise ValueError("ref_model_cls provided but ref_model_ckpt is None")
            self.ref_model_ckpt = Path(ref_model_ckpt).resolve()

        # TODO: Support moving the training model to CPU and the ref model to GPU while logging ref model diffs
        # I suspect Lightning will make this a bit of a pain so I haven't tried it yet.
        self.ref_model_device: torch.device = torch.device("cpu")

        self.__last_logged_step: int = -1
        self.__trainer: Trainer = None

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.__trainer = trainer

        if self.enabled:
            if self.ref_model is None and all((self.ref_model_ckpt, self.ref_model_cls)):
                if "vae" in self.ref_model_cls.lower():
                    logger.info(f"loading reference model from {self.ref_model_ckpt}")
                    self.ref_model = load_vae_ckpt(
                        self.ref_model_ckpt, asymmetric="asym" in self.ref_model_cls
                    )
                else:
                    raise NotImplementedError(f"ref_model_cls {self.ref_model_cls} is not implemented yet")

            if self.ref_model is not None:
                if self.ref_model.device != self.ref_model_device:
                    logger.info(f"moving reference model to {self.ref_model_device}")
                    self.ref_model = self.ref_model.to(self.ref_model_device)
                if self.ref_model.training:
                    logger.info("Freezing reference model")
                    self.ref_model = self.ref_model.eval().requires_grad_(False)

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

        fstem = f"gs{step:06d}_e{epoch:06d}_b{batch_idx:06d}"
        title = fstem.replace("-", "").replace("_", " ").upper()

        wandb_dict = {"trainer/global_step": step}
        table_dict = {}

        def add_to_both(k: str, v):
            wandb_dict[k] = v
            table_dict[k] = v

        if "samples" in images:
            samples = pt_to_pil(images.pop("samples"), aslist=True)
            for idx, img in enumerate(samples):
                img.save(save_dir / f"samples_{fstem}_{idx:02d}.png")

            if "caption" in batch:
                wandb_samples = [wandb.Image(img, caption=cap) for img, cap in zip(samples, batch["caption"])]
                try:
                    grid = self.make_caption_grid(samples, batch["caption"], title=title + " samples")
                    grid.save(save_dir.joinpath(f"samples_{fstem}_s-grid.png"))
                    wandb_dict[f"{split}/sample_grid"] = wandb.Image(grid, caption="Sample Grid")
                except Exception as e:
                    logger.exception("Failed to make sample grid, continuing", e)
            else:
                wandb_samples = [wandb.Image(img) for img in samples]
            add_to_both(f"{split}/samples", wandb_samples)

        for k in images:
            try:
                val = images[k]

                if isinstance(val[0], Tensor):
                    val = pt_to_pil(val, aslist=True)
                if isinstance(val, np.ndarray):
                    val = numpy_to_pil(val, aslist=True)

                # we should now have a list of PIL images, so save them
                if isinstance(val[0], Image.Image):
                    for idx, img in enumerate(val):
                        img.save(save_dir / f"{k.replace('/', '_')}_{fstem}_{idx:02d}.png")
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
                    for idx, img in enumerate(val):
                        img.save(save_dir / f"{k.replace('/', '_')}_{fstem}_{idx:02d}.png")
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
            if "vae" in self.ref_model_cls:
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


@torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=True)
@torch.no_grad()
def diff_images(
    inputs: Tensor,
    recons: Tensor,
    boost: float = 3.0,
) -> Tensor:
    diff = torch.clamp(recons, -1.0, 1.0).sub(inputs).abs().mul(0.5)

    boosted = diff.mul(boost).clamp(0.0, 1.0).mul(2.0).sub(1.0)
    diff = diff.mul(2.0).sub(1.0)

    return diff.contiguous(), boosted.contiguous()


@torch.no_grad()
def run_reference_vae(
    ref_model: AutoencoderKL | AsymmetricAutoencoderKL,
    inputs: Tensor,
    recons: Tensor,
    diff_boost: float = 3.0,
):
    inputs = inputs.detach().clone().to(ref_model.device, ref_model.dtype)
    recons = recons.detach().clone().to(ref_model.device, ref_model.dtype)

    # do these batch size 1 to reduce memory usage and compu
    ref_recons = torch.empty_like(recons).to(ref_model.device, ref_model.dtype)
    for idx, x in enumerate(inputs):
        ref_recons[idx] = ref_model.forward(x.unsqueeze(0)).sample.squeeze(0)

    diff_input, diff_input_boost = diff_images(inputs, ref_recons, diff_boost)
    diff_recons, diff_recons_boost = diff_images(ref_recons, recons, diff_boost)

    recon_mse = F.mse_loss(recons, inputs, reduction="none").mean(dim=(1, 2, 3))
    ref_recon_mse = F.mse_loss(ref_recons, inputs, reduction="none").mean(dim=(1, 2, 3))

    pass
