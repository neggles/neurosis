import logging
from contextlib import contextmanager
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Iterator, Literal, Optional
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

from neurosis.modules.autoencoding.losses import AutoencoderLoss
from neurosis.modules.ema import LitEma

from .utils import load_vae_ckpt

logger = logging.getLogger(__name__)


class DiffusersAutoencodingEngine(L.LightningModule):
    def __init__(
        self,
        model: str | PathLike | AutoencoderKL,
        loss: AutoencoderLoss = ...,
        optimizer: OptimizerCallable = ...,
        scheduler: LRSchedulerCallable = ...,
        input_key: str = "image",
        only_train_decoder: bool = False,
        diff_boost_factor: float = 3.0,
        wandb_watch: Optional[Literal["gradients", "parameters", "all"]] = None,
        wandb_watch_steps: int = -1,
        ema_decay: Optional[float] = None,
        ema_steps: int = -1,
        ema_kwargs: dict = {},
        ignore_keys: list[str] = [],
        compile_vae: bool = False,
        **model_kwargs,
    ):
        super().__init__()
        logger.info(f"Initializing {self.__class__.__name__}")

        # yeah, i shouldn't, but im gonna. one day i will work out why it is angry
        filterwarnings("ignore", message=r"^Grad strides do not match bucket view")

        self.input_key = input_key

        self.only_train_decoder = only_train_decoder
        self.diff_boost_factor = diff_boost_factor
        self.wandb_watch = wandb_watch
        self.wandb_watch_steps = wandb_watch_steps

        self.loss: AutoencoderLoss = loss

        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(model, nn.Module):
            self.vae: AutoencoderKL = model
            # make sure the model is in the right mode
            self.freeze(encoder=self.only_train_decoder, decoder=False)
        elif isinstance(model, (str, PathLike)):
            self._model_path = Path(model)
            self._model_kwargs = model_kwargs
            self.vae = load_vae_ckpt(self._model_path, **self._model_kwargs)
            # make sure the model is in the right mode
            self.freeze(encoder=self.only_train_decoder, decoder=False)
        else:
            raise ValueError(f"model must be nn.Module or str, got {model}")

        self.ema_steps = ema_steps
        self.ema_decay = ema_decay
        self.ema_kwargs = ema_kwargs
        self.use_ema = self.ema_decay is not None
        self.vae_ema: LitEma = None  # set up in configure_model

        self.compile_vae = compile_vae
        self.__compile_done = False

        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler", "loss"] + ignore_keys)

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
            self.vae_ema.store(self.parameters())
            self.vae_ema.copy_to(self)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.vae_ema.restore(self.parameters())
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
        return self.vae(sample, sample_posterior, return_dict, generator)

    def on_train_start(self):
        if self.wandb_watch_steps > 0:
            for wandb_logger in [x for x in self.trainer.loggers if isinstance(x, WandbLogger)]:
                wandb_logger.watch(self.vae, log=self.wandb_watch or "all", log_freq=self.wandb_watch_steps)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        inputs: Tensor = self.get_input(batch)

        recons: Tensor = self.forward(inputs, return_dict=False)[0]

        loss, log_dict = self.loss(inputs, recons, split="train", global_step=batch_idx)

        self.log_dict(log_dict, on_step=True, on_epoch=False)
        return loss.mean()

    def on_train_end(self) -> None:
        if self.wandb_watch_steps > 0:
            for wandb_logger in [x for x in self.trainer.loggers if isinstance(x, WandbLogger)]:
                wandb_logger.experiment.unwatch(self.vae)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.use_ema:
            if (
                self.global_step % (self.ema_steps * self.trainer.accumulate_grad_batches) == 0
                or self.trainer.is_last_batch
            ):
                self.vae_ema.update(self.parameters())

    def configure_model(self) -> None:
        # load the model if it's not already loaded
        if self.vae is None:
            self.vae = load_vae_ckpt(self._model_path, **self._model_kwargs)
            # make sure the model is in the right mode
            self.freeze(encoder=self.only_train_decoder, decoder=False)
        # set up EMA
        if self.use_ema is True and self.vae_ema is None:
            self.vae_ema = LitEma(self.vae, decay=self.ema_decay, **self.ema_kwargs)
            logger.info(f"Keeping EMAs of {len(list(self.vae_ema.buffers()))} weights.")

        # call configure_model() on loss if it exists
        if hasattr(self.loss, "configure_model"):
            self.loss.configure_model()

        if self.compile_vae and not self.__compile_done:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            self.vae = torch.compile(self.vae, dynamic=True, mode="reduce-overhead")
            self.__compile_done = True

    def configure_optimizers(self) -> OptimizerLRScheduler:
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
        split: str = "train",
        log_loss: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        inputs: Tensor = self.get_input(batch)[:num_img]
        recons = self.forward(inputs).sample
        diff = torch.clamp(recons, -1.0, 1.0).sub(inputs).abs().mul(0.5).clamp(0.0, 1.0)
        diff_boost = diff.mul(self.diff_boost_factor).clamp(0.0, 1.0)

        images_dict = {
            "inputs": inputs,
            "recons": recons,
            "diff": 2.0 * diff - 1.0,
            "diff_boost": 2.0 * diff_boost - 1.0,
        }

        if hasattr(self.loss, "log_images"):
            loss_images = self.loss.log_images(inputs, recons, split=split, **kwargs)
            images_dict.update(loss_images)

        if hasattr(self.loss, "log_loss") and log_loss is True:
            loss_log = self.loss.log_loss(inputs, recons, split=split, **kwargs)
            images_dict.update(loss_log)

        return images_dict
