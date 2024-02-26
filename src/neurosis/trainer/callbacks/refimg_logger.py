import logging
from os import PathLike
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import functional as F

from neurosis.models import DiffusersAutoencodingEngine
from neurosis.models.utils import load_vae_ckpt
from neurosis.trainer.common import BatchDictType, LogDictType, StepType, diff_images

from .image_logger import ImageLogger

logger = logging.getLogger(__name__)


class ReferenceData(BaseModel):
    inputs: Tensor
    recons: Tensor
    diff: Tensor
    diff_boost: Tensor
    mse_flt: Tensor
    mse_u8: Tensor
    ds_sim: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self.inputs = self.inputs.requires_grad_(False)
        self.recons = self.recons.requires_grad_(False)
        self.diff = self.diff.requires_grad_(False)
        self.diff_boost = self.diff_boost.requires_grad_(False)
        self.mse_flt = self.mse_flt.requires_grad_(False)
        self.mse_u8 = self.mse_u8.requires_grad_(False)
        self.ds_sim = self.ds_sim.requires_grad_(False)

        return super().model_post_init(__context)


class ReferenceModelImageLogger(ImageLogger):
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
        ref_cls: Optional[str] = None,
        ref_ckpt: Optional[str] = None,
        ref_device: str = "cpu",
        ref_compile: bool = False,
        ref_compile_opts: dict = {"mode": "reduce-overhead"},
        ref_data_path: Optional[PathLike] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            every_n_train_steps=every_n_train_steps,
            max_images=max_images,
            clamp=clamp,
            rescale=rescale,
            log_step_type=log_step_type,
            log_before_start=log_before_start,
            log_first_step=log_first_step,
            log_func_kwargs=log_func_kwargs,
            extra_log_keys=extra_log_keys,
            disabled=disabled,
            enable_autocast=enable_autocast,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            label_img=label_img,
            *args,
            **kwargs,
        )

        # load ref model if provided
        self.ref_model: AutoencoderKL = None
        self.ref_cls: Optional[str] = ref_cls
        self.ref_ckpt: Optional[Path] = Path(ref_ckpt) if ref_ckpt is not None else None
        self.ref_compile: bool = ref_compile
        self.ref_compile_opts: dict = ref_compile_opts

        if self.ref_cls is not None:
            if ref_ckpt is None:
                raise ValueError("ref_cls provided but ref_ckpt is None")
            self.ref_ckpt = Path(ref_ckpt).resolve()

        # TODO: Support moving the training model to CPU and the ref model to GPU while logging ref model diffs
        # I suspect Lightning will make this a bit of a pain so I haven't tried it yet.
        self.ref_device: torch.device = torch.device(ref_device)

        # for comparing MSE over time with a static test batch
        self.ref_data: Optional[ReferenceData] = None
        self.ref_data_path: Optional[Path] = Path(ref_data_path) if ref_data_path is not None else None
        if self.enabled:
            self.load_ref_model()
            if self.ref_data is None and self.ref_data_path is not None:
                payload = load_file(self.ref_data_path)
                self.ref_data = ReferenceData(**payload)

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)

    @rank_zero_only
    def load_ref_model(self):
        if self.ref_model is None and all((self.ref_ckpt, self.ref_cls)):
            if "vae" in self.ref_cls.lower():
                logger.info(f"loading reference model from {self.ref_ckpt}")
                self.ref_model = load_vae_ckpt(self.ref_ckpt).eval()
            else:
                raise NotImplementedError(f"ref_cls {self.ref_cls} is not implemented yet")

        if self.ref_model is not None:
            if self.ref_model.device != self.ref_device:
                logger.info(f"Moving reference model to {self.ref_device}")
                self.ref_model = self.ref_model.to(self.ref_device)
            if self.ref_device == torch.device("cpu"):
                logger.info("Casting reference model to float32 for CPU inference")
                if self.ref_model.dtype != torch.float32:
                    self.ref_model = self.ref_model.to(torch.float32)
            if self.ref_compile:
                logger.info("Compiling reference model")
                self.ref_model = torch.compile(self.ref_model, **self.ref_compile_opts)

            logger.info("Freezing reference model")
            self.ref_model = self.ref_model.requires_grad_(False)
            self.ref_model.set_attn_processor(AttnProcessor2_0())

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
        # hijack the log_local method to add our extra ref model images
        images = self.vae_reference_recons(images, num_img=self.max_images)
        images = self.vae_static_recons(pl_module, images, num_img=self.max_images)

        return super().log_local(split, images, batch, step, epoch, batch_idx, pl_module)

    @rank_zero_only
    @torch.no_grad()
    def vae_static_recons(
        self,
        pl_module: DiffusersAutoencodingEngine,
        images: LogDictType,
        num_img: int = 1,
        split: str = "train",
        **kwargs,
    ) -> LogDictType:
        if self.ref_model is None or self.ref_data is None:
            return images

        inputs = self.ref_data.inputs[:num_img].clone()
        log_dict = self.call_log_images(
            pl_module,
            batch={pl_module.input_key: inputs.detach().to(pl_module.device, pl_module.dtype)},
            split="static",
            num_img=len(inputs),
            log_loss=True,
        )

        recons: Tensor = log_dict.get("recons").cpu()
        diff_input: Tensor = log_dict.get("diff").cpu()
        diff_input_boost: Tensor = log_dict.get("diff_boost").cpu()

        vae_mse = F.mse_loss(recons, inputs, reduction="mean").mul(65025.0).cpu()
        ref_mse = self.ref_data.mse_flt.mean().cpu()

        # percentage decrease in MSE from ref to student
        pct_decrease = (vae_mse - ref_mse) / ref_mse * -1

        # diff images against ref reconstructions
        diff_ref, diff_ref_boost = diff_images(self.ref_data.recons, recons, self.diff_boost)

        images_dict = {
            "static/inputs": inputs.contiguous(),
            "static/recons": recons.contiguous(),
            "static/diff_input": diff_input.contiguous(),
            "static/diff_input_boost": diff_input_boost.contiguous(),
            "static/diff_ref": diff_ref.contiguous(),
            "static/diff_ref_boost": diff_ref_boost.contiguous(),
            "static/mse_flt": vae_mse.item(),
            "static/mse_pct": pct_decrease.item(),
        }
        loss_dict = {
            k.replace("loss/", "loss_"): v for k, v in log_dict.items() if k.startswith("static/loss")
        }
        images_dict.update(loss_dict)
        images.update(images_dict)
        return images

    @rank_zero_only
    @torch.no_grad()
    def vae_reference_recons(
        self,
        images: LogDictType,
        num_img: int = 1,
        prefix: str = "ref",
        **kwargs,
    ) -> LogDictType:
        if self.ref_model is None:
            return images

        inputs: Tensor = images.get("inputs")[:num_img]
        recons: Tensor = images.get("recons")[:num_img]

        inputs = inputs.detach().clone().to(self.ref_device, self.ref_model.dtype)
        recons = recons.detach().clone().to(self.ref_device, self.ref_model.dtype)

        # do these batch size 1 to reduce memory usage and compu
        ref_recons = torch.empty_like(recons, requires_grad=False).to(self.ref_device)
        for idx, x in enumerate(inputs):
            ref_recons[idx] = self.ref_model.forward(x.unsqueeze(0)).sample.squeeze(0)

        diff_input, _ = diff_images(inputs, ref_recons, self.diff_boost)
        diff_ref, diff_ref_boost = diff_images(ref_recons, recons, self.diff_boost)
        ref_mse = F.mse_loss(ref_recons, inputs, reduction="mean") * 65025.0

        images_dict = {
            f"{prefix}/recons": ref_recons,
            f"{prefix}/diff_input": diff_input,
            f"{prefix}/mse_flt": ref_mse.item(),
            f"{prefix}/diff": diff_ref,
            f"{prefix}/diff_boost": diff_ref_boost,
        }
        images.update(images_dict)
        return images
