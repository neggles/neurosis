import logging
from contextlib import contextmanager
from math import ceil
from os import PathLike
from pathlib import Path
from typing import Optional

import lightning.pytorch as L
import numpy as np
import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers.wandb import WandbLogger
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.models.autoencoder import AutoencodingEngine
from neurosis.modules.diffusion import (
    BaseDiffusionSampler,
    Denoiser,
    DiffusionLoss,
    UNetModel,
)
from neurosis.modules.diffusion.hooks import LossHook
from neurosis.modules.diffusion.wrappers import OpenAIWrapper
from neurosis.modules.ema import LitEma
from neurosis.modules.encoders import GeneralConditioner
from neurosis.modules.encoders.embedding import AbstractEmbModel
from neurosis.utils import disabled_train, log_txt_as_img, np_text_decode

logger = logging.getLogger(__name__)


class DiffusionEngine(L.LightningModule):
    def __init__(
        self,
        model: UNetModel,
        denoiser: Denoiser,
        first_stage_model: AutoencodingEngine,
        conditioner: GeneralConditioner,
        sampler: Optional[BaseDiffusionSampler],
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
        loss_fn: Optional[DiffusionLoss],
        ckpt_path: Optional[PathLike] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast: bool = False,
        input_key: str = "jpg",
        log_keys: Optional[list] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        vae_batch_size: Optional[int] = None,
        forward_hooks: list[LossHook] = [],
    ):
        super().__init__()
        logger.info("Initializing DiffusionEngine")

        self.log_keys = log_keys
        self.input_key = input_key

        self.model: UNetModel = OpenAIWrapper(model, compile_model=compile_model)
        self.denoiser = denoiser
        self.sampler = sampler
        self.conditioner = conditioner

        self.optimizer = optimizer
        self.scheduler = scheduler

        # do first stage model setup
        logger.info("Loading first stage (VAE) model...")
        self._init_first_stage(first_stage_model)

        self.loss_fn = loss_fn
        self.forward_hooks = forward_hooks

        self.use_ema = use_ema
        if self.use_ema:
            logger.info("Using EMA")
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        else:
            self.model_ema = None

        self.scale_factor: float = scale_factor
        self.first_stage_autocast: bool = not disable_first_stage_autocast
        self.no_cond_log: bool = no_cond_log
        self.vae_batch_size = vae_batch_size

        if ckpt_path is not None:
            self.init_from_ckpt(Path(ckpt_path))

        self.save_hyperparameters(
            ignore=[
                "model",
                "denoiser",
                "first_stage_model",
                "conditioner",
                "sampler",
                "loss_fn",
                "optimizer",
                "scheduler",
            ]
        )
        for pl_logger in self.loggers:
            if isinstance(pl_logger, WandbLogger):
                pl_logger.experiment.config.update(self.hparams)

    def init_from_ckpt(self, path: Path) -> None:
        if path.suffix == ".safetensors":
            sd = load_safetensors(path)
        elif path.suffix in CHECKPOINT_EXTNS:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise NotImplementedError(f"Unknown checkpoint extension {path.suffix}")

        missing, unexpected = self.load_state_dict(sd, strict=False)
        logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            logger.warn(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, model: AutoencodingEngine):
        model = model.eval()
        model.freeze()
        model.train = disabled_train
        self.first_stage_model = model

    def get_input(self, batch: dict[str, Tensor]) -> Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        inputs: Tensor = batch[self.input_key]
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        return inputs

    @torch.no_grad()
    def decode_first_stage(self, z: Tensor) -> Tensor:
        z = 1.0 / self.scale_factor * z
        n_samples = self.vae_batch_size or z.shape[0]

        n_rounds = ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast(self.device.type, enabled=self.first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.decode(z[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x: Tensor) -> Tensor:
        n_samples = self.vae_batch_size or x.shape[0]
        n_rounds = ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast(self.device.type, enabled=self.first_stage_autocast):
            for n in range(n_rounds):
                out = self.first_stage_model.encode(x[n * n_samples : (n + 1) * n_samples])
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z

    def forward(self, x: Tensor, batch: dict[str, Tensor]) -> Tensor:
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        # run any pre-step hooks
        for hook in self.forward_hooks:
            hook.pre_hook(self.trainer, self, batch, batch_idx)

        # get inputs and encode them
        inputs = self.get_input(batch)
        latents = self.encode_first_stage(inputs)

        # add global step to batch
        batch["global_step"] = self.global_step
        # run the actual step
        loss = self(latents, batch)

        # run any post-step hooks
        loss_dict = {}
        for hook in self.forward_hooks:
            loss, loss_dict = hook(self, batch, loss, loss_dict)

        # log the adjusted loss
        loss_dict.update({"train/loss": loss.detach().clone().mean()})

        self.log_dict(loss_dict, prog_bar=True, on_step=True, on_epoch=False)
        return loss.mean()

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def configure_optimizers(self):
        param_groups = []
        unet_params = {"name": "UNet", "params": list(self.model.parameters())}
        # add initial_lr if set on the unet
        if hasattr(self.model, "base_lr") and self.model.base_lr is not None:
            logger.info(f"Setting initial_lr for unet to {self.model.base_lr:.2e}")
            unet_params["initial_lr"] = self.model.base_lr
        param_groups.append(unet_params)

        embedder: AbstractEmbModel
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                embedder_name = getattr(embedder, "name", embedder.__class__.__name__)
                logger.info(f"Adding {embedder_name} to trainable parameter groups")

                embedder_params = {"name": embedder_name, "params": list(embedder.parameters())}
                # add initial_lr if set on the embedder
                if hasattr(embedder, "base_lr") and embedder.base_lr is not None:
                    logger.info(f"Setting initial_lr for {embedder_name} to {embedder.base_lr:.2e}")
                    embedder_params["initial_lr"] = embedder.base_lr

                param_groups.append(embedder_params)

        optimizer = self.optimizer(param_groups)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        uc: Optional[dict] = None,
        batch_size: int = 4,
        shape: Optional[tuple | list] = None,
        **model_kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        def denoiser_cb(inputs: Tensor, sigma: Tensor, c: dict) -> Tensor:
            return self.denoiser(self.model, inputs, sigma, c, **model_kwargs)

        samples = self.sampler(denoiser_cb, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: dict[str, Tensor], num_img: int, split: str = "train") -> dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        log_dict = dict()
        if self.no_cond_log is True:
            return log_dict

        wh = batch[self.input_key].shape[2:]

        embedder: AbstractEmbModel
        for embedder in self.conditioner.embedders:
            if (self.log_keys is None) or (embedder.input_key in self.log_keys):
                inputs = batch[embedder.input_key][:num_img]
                if isinstance(inputs, Tensor):
                    if inputs.dim() == 1:
                        # class-conditional, convert integer to string
                        inputs = [str(inputs[i].item()) for i in range(inputs.shape[0])]
                        value = log_txt_as_img(wh, inputs, size=min(wh[0] // 4, 96))
                    elif inputs.dim() == 2:
                        # size and crop cond and the like
                        log_strings = [
                            "x".join(str(x) for x in inputs[i].tolist()) for i in range(inputs.shape[0])
                        ]
                        value = log_txt_as_img(wh, log_strings, size=min(wh[0] // 4, 96))
                    else:
                        raise NotImplementedError("Tensor conditioning with dim > 2 not implemented")

                elif isinstance(inputs, list):
                    if isinstance(inputs[0], (bytes, np.bytes_)):
                        inputs = np_text_decode(inputs, aslist=True)
                    if isinstance(inputs[0], str):
                        # strings
                        value = log_txt_as_img(wh, inputs, size=min(wh[0] // 20, 24))
                    else:
                        raise NotImplementedError(f"Can't log conditioning for list[{type(inputs[0])}]")

                else:
                    raise NotImplementedError(f"Can't log conditioning for {type(inputs)}")
                log_dict[embedder.input_key] = value
        return log_dict

    @torch.no_grad()
    def log_images(
        self,
        batch: dict,
        num_img: int = 4,
        split: str = "train",
        sample: bool = True,
        ucg_keys: list[str] = None,
        **kwargs,
    ) -> dict:
        inputs: Tensor = self.get_input(batch)[:num_img]
        num_img = len(inputs)

        input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys is None or len(ucg_keys) == 0:
            ucg_keys = input_keys

        else:
            if any(x not in input_keys for x in ucg_keys):
                raise ValueError(
                    "Each defined ucg key for sampling must be in the provided conditioner input keys!"
                    f"\nRequested UCG keys: {ucg_keys}" + f"\nAvailable input keys: {input_keys}"
                )

        # log inputs and VAE reconstructions
        latents: Tensor = self.encode_first_stage(inputs)
        recons = self.decode_first_stage(latents)

        images_dict = {
            "inputs": inputs.cpu(),
            "recons": recons.cpu(),
        }

        # log conditioning
        cond_dict = self.log_conditionings(batch, num_img, split)
        images_dict.update(cond_dict)

        force_uc_zero = ucg_keys if len(self.conditioner.embedders) > 0 else []
        cond, uncond = self.conditioner.get_unconditional_conditioning(
            batch, force_uc_zero_embeddings=force_uc_zero
        )
        for key in cond:
            if isinstance(cond[key], Tensor):
                cond[key] = cond[key][:num_img].to(self.device)
                uncond[key] = uncond[key][:num_img].to(self.device)

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(
                    cond=cond, shape=latents.shape[1:], uc=uncond, batch_size=num_img, **kwargs
                )
            samples = self.decode_first_stage(samples)
            images_dict["samples"] = samples.cpu()

        return images_dict
