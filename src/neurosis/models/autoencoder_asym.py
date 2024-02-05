import logging
from contextlib import contextmanager
from functools import cached_property
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional
from warnings import filterwarnings

import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from lightning import pytorch as L
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Generator, Tensor, nn

from neurosis.modules.autoencoding.asymmetric import AsymmetricAutoencoderKL
from neurosis.modules.autoencoding.losses import (
    AutoencoderLPIPSWithDiscr,
    AutoencoderPerceptual,
    GeneralLPIPSWithDiscriminator,
)
from neurosis.modules.distributions import DiagonalGaussianDistribution
from neurosis.modules.ema import LitEma
from neurosis.trainer.util import EMATracker

from .utils import load_vae_ckpt

logger = logging.getLogger(__name__)


class AsymmetricAutoencodingEngine(L.LightningModule):
    def __init__(
        self,
        model: str | PathLike | AutoencoderKL | AsymmetricAutoencoderKL,
        optimizer: OptimizerCallable = ...,
        scheduler: LRSchedulerCallable = ...,
        loss: Optional[AutoencoderPerceptual | nn.Module] = None,
        input_key: str = "image",
        asymmetric: bool = False,
        only_train_decoder: bool = False,
        ema_decay: Optional[float] = None,
        diff_boost_factor: float = 3.0,
        wandb_watch: int = -1,
        ignore_keys: list[str] = [],
        **model_kwargs,
    ):
        super().__init__()
        logger.info(f"Initializing {self.__class__.__name__}")

        # yeah, i shouldn't, but im gonna. one day i will work out why it is angry
        filterwarnings("ignore", message=r"^Grad strides do not match bucket view")

        self.input_key = input_key

        self.use_ema = ema_decay is not None
        self.ema_decay = ema_decay
        self.model_ema: LitEma = None

        self.only_train_decoder = only_train_decoder
        self.asymmetric = asymmetric
        self.diff_boost_factor = diff_boost_factor
        self.wandb_watch = wandb_watch

        self.loss: AutoencoderPerceptual = loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(model, nn.Module):
            self.vae: AutoencoderKL | AsymmetricAutoencoderKL = model
            self.freeze(encoder=self.only_train_decoder, decoder=False)
        elif isinstance(model, (str, PathLike)):
            self._model_path = Path(model)
            self._model_kwargs = model_kwargs
            self.vae: AutoencoderKL | AsymmetricAutoencoderKL = None
        else:
            raise ValueError(f"model must be nn.Module or str, got {model}")

        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler", "loss"] + ignore_keys)

    def configure_model(self) -> None:
        if self.vae is None:
            self.vae = load_vae_ckpt(self._model_path, asymmetric=self.asymmetric, **self._model_kwargs)
            self.freeze(encoder=self.only_train_decoder, decoder=False)

        if self.model_ema is None and self.use_ema:
            self.model_ema = LitEma(self.vae, decay=self.ema_decay)
            logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))} weights.")

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        sample: Tensor = batch[self.input_key]
        if sample.ndim == 3:
            sample = sample.unsqueeze(0)
        return sample

    @contextmanager
    def ema_scope(self, context: str = None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        return chain(self.vae.encoder.parameters(), self.vae.quant_conv.parameters())

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        if self.loss is not None and hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            return chain(
                self.vae.post_quant_conv.parameters(),
                self.vae.decoder.parameters(),
                self.loss.get_trainable_autoencoder_parameters(),
            )
        return chain(self.vae.post_quant_conv.parameters(), self.vae.decoder.parameters())

    def get_last_layer(self) -> Tensor:
        return self.vae.decoder.conv_out.weight

    def freeze(self, encoder: bool = True, decoder: bool = True):
        if encoder is True and decoder is True:
            logger.debug("Freezing all parameters")
            self.vae.requires_grad_(False)
        elif isinstance(encoder, bool) and isinstance(decoder, bool):
            for p in self.get_encoder_params():
                p.requires_grad_(not encoder)
            for p in self.get_decoder_params():
                p.requires_grad_(not decoder)
        else:
            raise ValueError("encoder and decoder must be bool or None")

    def encode(self, x: Tensor, return_dict: bool = True) -> AutoencoderKLOutput | Tensor:
        return self.vae.encode(x, return_dict)

    def decode(self, z: Tensor, return_dict: bool = True) -> DecoderOutput | Tensor:
        return self.vae.decode(z, return_dict)

    def forward(
        self,
        sample: Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[Generator] = None,
    ) -> DecoderOutput | Tensor:
        return self.vae.forward(sample, sample_posterior, return_dict, generator)

    def on_train_start(self):
        if self.wandb_watch > 0:
            for wandb_logger in [x for x in self.trainer.loggers if isinstance(x, WandbLogger)]:
                wandb_logger.watch(self.vae, log="all", log_freq=self.wandb_watch)

    def on_train_end(self) -> None:
        if self.wandb_watch > 0:
            for wandb_logger in [x for x in self.trainer.loggers if isinstance(x, WandbLogger)]:
                wandb_logger.experiment.unwatch(self.vae)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        inputs: Tensor = self.get_input(batch)

        recons: Tensor = self.forward(inputs).sample

        loss, log_dict = self.loss(inputs, recons, split="train", global_step=batch_idx)

        self.log_dict(log_dict, on_step=True, on_epoch=False)
        return loss.mean()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.only_train_decoder:
            logger.info("Freezing encoder!")
            self.freeze(encoder=True, decoder=False)

        encoder_params = {
            "name": "Encoder",
            "params": list(self.get_encoder_params()),
        }
        decoder_params = {
            "name": "Decoder",
            "params": list(self.get_decoder_params()),
        }

        if self.only_train_decoder:
            optimizer = self.optimizer([decoder_params])
        else:
            optimizer = self.optimizer([encoder_params, decoder_params])

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return optimizer

    @torch.no_grad()
    def log_images(
        self,
        batch: dict,
        num_img: int = 1,
        **kwargs,
    ) -> dict:
        inputs: Tensor = self.get_input(batch)[:num_img]

        with torch.inference_mode():
            recons = self.forward(inputs).sample
            diff = torch.clamp(recons, -1.0, 1.0).sub(inputs).abs().mul(0.5).clamp(0.0, 1.0)
            diff_boost = diff.mul(self.diff_boost_factor).clamp(0.0, 1.0)

        log_dict = {
            "inputs": inputs,
            "recons": recons,
            "diff": 2.0 * diff - 1.0,
            "diff_boost": 2.0 * diff_boost - 1.0,
        }

        if hasattr(self.loss, "log_images"):
            loss_log = self.loss.log_images(inputs, recons, **kwargs)
            log_dict.update(loss_log)

        return log_dict


class AsymmetricAutoencodingEngineDisc(AsymmetricAutoencodingEngine):
    def __init__(
        self,
        *args,
        loss: GeneralLPIPSWithDiscriminator | AutoencoderLPIPSWithDiscr,
        accumulate_grad_batches: int = 1,
        **kwargs,
    ):
        self.automatic_optimization = False
        super().__init__(loss=loss, ignore_keys=["loss"], *args, **kwargs)

        self.loss_ema_disc = EMATracker(alpha=self.loss_ema.alpha)
        self.acc_grad_batches = accumulate_grad_batches

    @property
    def accumulate_grad_batches(self) -> int:
        return self.acc_grad_batches

    @cached_property
    def disc_start(self) -> int:
        if hasattr(self.loss, "disc_start"):
            return self.loss.disc_start
        else:
            return -1

    def get_discriminator_params(self) -> Iterator[nn.Parameter]:
        if hasattr(self.loss, "get_trainable_parameters"):
            yield from self.loss.get_trainable_parameters()  # e.g., discriminator
        else:
            yield from ()

    def get_opt_idx(self, opts: list[OptimizerLRScheduler], batch_idx: int) -> int:
        """Get the active optimizer index for this batch"""
        if (batch_idx < self.disc_start) or (self.disc_start < 0):
            # force optimizer zero if we're not yet at the disc_start step
            return 0
        else:
            # otherwise switch between AE and disc optimizers every acc_grad_batches
            return batch_idx % (self.acc_grad_batches * len(opts)) // self.acc_grad_batches

    def check_grad_accumulation(self, batch_idx: int) -> bool:
        """Check if we're ready to step the optimizer (accumulation done or last step)"""
        return ((batch_idx + 1) % self.acc_grad_batches == 0) or self.trainer.is_last_batch

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        opts = self.optimizers()
        scheds = self.lr_schedulers()

        if not isinstance(opts, list):
            opts = [opts]
        if not isinstance(scheds, list):
            scheds = [scheds]

        optimizer_idx = self.get_opt_idx(opts, batch_idx)
        accumulated_grads = self.check_grad_accumulation(batch_idx)

        if optimizer_idx == 0:
            opt_ae, sched_ae = opts[0], scheds[0]

            def closure_ae():
                loss, log_dict = self.inner_training_step(batch, batch_idx, optimizer_idx)
                loss = loss / self.acc_grad_batches
                self.manual_backward(loss)
                return loss, log_dict

            with opt_ae.toggle_model(sync_grad=accumulated_grads):
                if accumulated_grads:
                    loss, log_dict = opt_ae.step(closure=closure_ae)
                    sched_ae.step()
                    opt_ae.zero_grad()
                else:
                    loss, log_dict = closure_ae()

        elif optimizer_idx == 1:
            opt_disc, sched_disc = opts[1], scheds[1]

            def closure_disc():
                loss, log_dict = self.inner_training_step(batch, batch_idx, optimizer_idx)
                loss = loss.div(self.acc_grad_batches)
                self.manual_backward(loss)
                return loss, log_dict

            with opt_disc.toggle_model(sync_grad=accumulated_grads):
                if accumulated_grads:
                    loss, log_dict = opt_disc.step(closure=closure_disc)
                    sched_disc.step()
                    opt_disc.zero_grad()
                else:
                    loss, log_dict = closure_disc()

        log_dict["optimizer_idx"] = optimizer_idx
        self.log_dict(log_dict, sync_dist=accumulated_grads, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        ae_loss, log_dict = self.inner_validation_step(batch, batch_idx)
        with self.ema_scope():
            ae_loss_ema, log_dict_ema = self.inner_validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)

        self.log_dict(log_dict, sync_dist=True, on_step=True, on_epoch=False)
        return ae_loss.mean() + ae_loss_ema.mean() / 2

    def inner_validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> dict:
        x = self.get_input(batch)

        posterior: DiagonalGaussianDistribution = self.vae.encode(x).latent_dist
        z = posterior.mode()
        recons = self.vae.decode(z).sample

        # autoencoder loss
        loss_kwargs = dict(
            inputs=x,
            reconstructions=recons,
            global_step=batch_idx,
            split=f"val{postfix}",
        )
        if "last_layer" in self.loss.forward_keys:
            loss_kwargs["last_layer"] = self.get_last_layer()

        ae_loss, log_dict = self.loss(optimizer_idx=0, **loss_kwargs)

        if len(self.optimizers()) > 1:
            _, log_dict_disc = self.loss(optimizer_idx=1, **loss_kwargs)
            log_dict.update(log_dict_disc)

        return ae_loss, log_dict

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.only_train_decoder:
            logger.info("Freezing encoder!")
            self.freeze(encoder=True, decoder=False)

        encoder_params = {
            "name": "encoder",
            "params": list(self.get_encoder_params()),
        }
        decoder_params = {
            "name": "decoder",
            "params": list(self.get_decoder_params()),
        }

        if self.only_train_decoder:
            opt_ae = self.optimizer([decoder_params])
        else:
            opt_ae = self.optimizer([encoder_params, decoder_params])

        # set up for autoencoder
        sched_ae = self.scheduler(opt_ae)
        opt_sched_ae = {
            "optimizer": opt_ae,
            "lr_scheduler": {"scheduler": sched_ae, "interval": "step"},
        }

        opt_sched_disc = None
        disc_params = list(self.get_discriminator_params())
        if len(disc_params) > 0:
            disc_params = {"name": "discriminator", "params": disc_params}
            opt_disc = self.optimizer([disc_params])
            sched_disc = self.scheduler(opt_disc)
            opt_sched_disc = {
                "optimizer": opt_disc,
                "lr_scheduler": {"scheduler": sched_disc, "interval": "step"},
            }

        if opt_sched_disc is not None:
            return [opt_sched_ae, opt_sched_disc]
        else:
            return [opt_sched_ae]
