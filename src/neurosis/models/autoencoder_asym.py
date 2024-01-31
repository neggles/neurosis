import logging
from contextlib import contextmanager
from functools import cached_property, wraps
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional
from warnings import warn

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from lightning import pytorch as L
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Generator, Tensor, nn
from torch.optim.optimizer import Optimizer

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.modules.autoencoding.asymmetric import AsymmetricAutoencoderKL
from neurosis.modules.autoencoding.losses import AutoencoderLPIPSWithDiscr, GeneralLPIPSWithDiscriminator
from neurosis.modules.distributions import DiagonalGaussianDistribution
from neurosis.modules.ema import LitEma
from neurosis.trainer.util import EMATracker

logger = logging.getLogger(__name__)


class AsymmetricAutoencodingEngine(L.LightningModule):
    def __init__(
        self,
        model: str | PathLike | AutoencoderKL | AsymmetricAutoencoderKL,
        optimizer: OptimizerCallable = ...,
        scheduler: LRSchedulerCallable = ...,
        loss: str | nn.Module = "mse",
        input_key: str = "image",
        log_keys: Optional[list] = None,
        asymmetric: bool = False,
        only_train_decoder: bool = False,
        base_lr: Optional[float] = None,
        ema_decay: Optional[float] = None,
        loss_ema_alpha: float = 0.02,
        diff_boost_factor: float = 3.0,
        ignore_keys: list[str] = [],
        **model_kwargs,
    ):
        super().__init__()
        logger.info(f"Initializing {self.__class__.__name__}")

        self.input_key = input_key
        self.log_keys = log_keys

        self.base_lr = base_lr
        self.use_ema = ema_decay is not None
        self.ema_decay = ema_decay
        self.model_ema: LitEma = None

        self.only_train_decoder = only_train_decoder
        self.asymmetric = asymmetric
        self.loss_ema = EMATracker(alpha=loss_ema_alpha)
        self.diff_boost_factor = diff_boost_factor

        self.loss = loss

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

        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler", "key_info_path"] + ignore_keys)

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
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            return chain(
                self.vae.post_quant_conv.parameters(),
                self.vae.decoder.parameters(),
                self.loss.get_trainable_autoencoder_parameters(),
            )
        return chain(self.vae.post_quant_conv.parameters(), self.vae.decoder.parameters())

    def freeze(self, encoder: bool = True, decoder: bool = True):
        if isinstance(encoder, bool):
            self.vae.encoder.requires_grad_(not encoder)
            self.vae.quant_conv.requires_grad_(not encoder)
        else:
            raise ValueError(f"encoder must be bool, got {encoder}")

        if isinstance(decoder, bool):
            self.vae.decoder.requires_grad_(not decoder)
            self.vae.post_quant_conv.requires_grad_(not decoder)
        else:
            raise ValueError(f"decoder must be bool, got {decoder}")

    @wraps(AutoencoderKL.encode)
    def encode(self, x: Tensor, return_dict: bool = True) -> AutoencoderKLOutput | Tensor:
        return self.vae.encode(x, return_dict)

    @wraps(AutoencoderKL.decode)
    def decode(self, z: Tensor, return_dict: bool = True) -> DecoderOutput | Tensor:
        return self.vae.decode(z, return_dict)

    @wraps(AutoencoderKL.forward)
    def forward(
        self,
        sample: Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[Generator] = None,
    ) -> DecoderOutput | Tensor:
        return self.vae.forward(sample, sample_posterior, return_dict, generator)

    def get_loss(self, recons: Tensor, target: Tensor, kind: Optional[str] = None, *args, **kwargs) -> Tensor:
        kind = self.loss if kind is None else kind
        if not isinstance(kind, str):
            warn(f"loss type {self.loss} is not a string, assuming it's a loss function")
            return self.loss(recons, target, *args, **kwargs)

        match kind:
            case "l1":
                return F.l1_loss(recons.contiguous(), target.contiguous(), reduction="mean")
            case "l2" | "mse":
                return F.mse_loss(recons.contiguous(), target.contiguous(), reduction="mean")
            case "nll":
                return F.nll_loss(recons.contiguous(), target.contiguous(), reduction="mean")
            case nn.Module():
                return self.loss(recons.contiguous(), target.contiguous(), *args, **kwargs)
            case _:
                raise ValueError(f"loss type {self.loss} not supported")

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x: Tensor = self.get_input(batch)

        posterior: DiagonalGaussianDistribution = self.vae.encode(x).latent_dist
        z: Tensor = posterior.mode()

        recon: Tensor = self.vae.decode(z).sample
        ae_loss = self.get_loss(recon, x)

        # update EMA loss tracker
        log_loss = ae_loss.detach().cpu().item()
        self.loss_ema.update(log_loss)

        self.log_dict(
            {
                "train/loss": log_loss,
                "train/loss_ema": self.loss_ema.value,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=x.shape[0],
        )
        return ae_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        param_groups = []
        encoder_params = {
            "name": "Encoder",
            "params": list(self.get_encoder_params()),
        }

        if self.only_train_decoder:
            logger.info("Freezing encoder!")
            for param in encoder_params["params"]:
                param.requires_grad_(False)
        else:
            param_groups.append(encoder_params)

        decoder_params = {
            "name": "Decoder",
            "params": list(self.get_decoder_params()),
        }
        param_groups.append(decoder_params)

        if self.base_lr is not None:
            for param_group in param_groups:
                param_group["initial_lr"] = self.base_lr

        optimizer = self.optimizer(param_groups)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # torch.cuda.empty_cache()
        return super().on_before_optimizer_step(optimizer)

    @torch.no_grad()
    def log_images(
        self,
        batch: dict,
        num_img: int = 1,
        sample: bool = True,
        ucg_keys: list[str] = None,
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


def load_vae_ckpt(
    model_path: Path, asymmetric: bool = False, **model_kwargs
) -> AsymmetricAutoencoderKL | AutoencoderKL:
    model_cls = AsymmetricAutoencoderKL if asymmetric else AutoencoderKL

    if model_path.is_file():
        if model_path.suffix.lower() in CHECKPOINT_EXTNS:
            load_fn = model_cls.from_single_file
        else:
            raise ValueError(f"model file {model_path} is not a valid checkpoint file")
    elif model_path.is_dir():
        if model_path.joinpath("config.json").exists():
            load_fn = model_cls.from_pretrained
        else:
            raise ValueError(f"model folder {model_path} is not a HF checkpoint (no config.json)")
    else:
        raise ValueError(f"model path {model_path} is not a file or directory")

    return load_fn(model_path, **model_kwargs)


def is_encoder_key(str):
    if str.startswith("encoder") or str.startswith("quant_conv"):
        return True
    return False


class AsymmetricAutoencodingEngineDisc(AsymmetricAutoencodingEngine):
    def __init__(
        self,
        *args,
        loss: GeneralLPIPSWithDiscriminator | AutoencoderLPIPSWithDiscr,
        accumulate_grad_batches: int = 1,
        wandb_watch: int = -1,
        **kwargs,
    ):
        super().__init__(loss="mse", ignore_keys=["loss"], *args, **kwargs)
        self.automatic_optimization = False

        self.loss: GeneralLPIPSWithDiscriminator = loss
        self.loss_ema_disc = EMATracker(alpha=self.loss_ema.alpha)
        self.disc_start = self.loss.disc_start

        self.acc_grad_batches = accumulate_grad_batches
        self.wandb_watch = wandb_watch

    def on_train_start(self):
        if self.wandb_watch > 0:
            for wandb_logger in [x for x in self.trainer.loggers if isinstance(x, WandbLogger)]:
                wandb_logger.watch(self.vae, log="all", log_freq=self.wandb_watch)

    def on_train_end(self) -> None:
        if self.wandb_watch > 0:
            for wandb_logger in [x for x in self.trainer.loggers if isinstance(x, WandbLogger)]:
                wandb_logger.experiment.unwatch(self.vae)

    @cached_property
    def disc_start(self) -> int:
        return self.loss.disc_start

    def get_discriminator_params(self) -> Iterator[nn.Parameter]:
        if hasattr(self.loss, "get_trainable_parameters"):
            yield from self.loss.get_trainable_parameters()  # e.g., discriminator
        else:
            yield from ()

    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

    def inner_training_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int = 0,
        optimizer_idx: int = 0,
    ) -> Tensor:
        x = self.get_input(batch)

        posterior: DiagonalGaussianDistribution = self.vae.encode(x).latent_dist
        z = posterior.mode()
        recons = self.vae.decode(z).sample

        loss_kwargs = dict(
            inputs=x.contiguous(),
            reconstructions=recons.contiguous(),
            global_step=self.global_step,
            optimizer_idx=optimizer_idx,
            split="train",
        )
        if "last_layer" in self.loss.forward_keys:
            loss_kwargs["last_layer"] = self.get_last_layer()

        if optimizer_idx == 0:
            # autoencoder loss
            ae_loss, log_dict_ae = self.loss(**loss_kwargs)
            return ae_loss.mean(), log_dict_ae
        elif optimizer_idx == 1:
            # discriminator
            disc_loss, log_dict_disc = self.loss(**loss_kwargs)
            return disc_loss.mean(), log_dict_disc
        else:
            raise ValueError(f"Unknown optimizer ID {optimizer_idx}")

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        opts = self.optimizers()
        scheds = self.lr_schedulers()

        if not isinstance(opts, list):
            opts = [opts]
        if not isinstance(scheds, list):
            scheds = [scheds]

        if self.global_step < self.disc_start:
            # force optimizer zero if we're not yet at the disc_start step
            optimizer_idx = 0
        else:
            # otherwise switch between AE and disc optimizers every acc_grad_batches
            optimizer_idx = self.global_step % (self.acc_grad_batches * len(opts)) // self.acc_grad_batches

        # have we finished accumulating gradients this batch?
        accumulated_grads = (self.global_step + 1) % self.acc_grad_batches == 0

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

            self.loss_ema.update(log_dict["train/loss/total"].cpu().item())
            log_dict["train/loss/total_ema"] = self.loss_ema.value

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

            self.loss_ema_disc.update(log_dict["train/loss/disc"].cpu().item())
            log_dict["train/loss/disc_ema"] = self.loss_ema_disc.value

        if accumulated_grads:
            self.log_dict(log_dict, sync_dist=optimizer_idx == 1, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> dict:
        x = self.get_input(batch)

        posterior: DiagonalGaussianDistribution = self.vae.encode(x).latent_dist
        z = posterior.mode()
        kl_loss = posterior.kl()
        reg_log = {"kl_loss": torch.sum(kl_loss) / kl_loss.shape[0]}
        recons = self.vae.decode(z).sample

        # autoencoder loss
        loss_kwargs = dict(
            inputs=x,
            reconstructions=recons,
            global_step=self.global_step,
            split=f"val{postfix}",
        )
        if "last_layer" in self.loss.forward_keys:
            loss_kwargs["last_layer"] = self.get_last_layer()

        ae_loss, log_dict = self.loss(optimizer_idx=0, **loss_kwargs)

        if len(self.optimizers()) > 1:
            disc_loss, log_dict_disc = self.loss(optimizer_idx=1, **loss_kwargs)
            log_dict.update(log_dict_disc)

        self.log_dict(log_dict, sync_dist=True)
        return log_dict

    def configure_optimizers(self) -> OptimizerLRScheduler:
        param_groups = []

        encoder_params = {
            "name": "Encoder",
            "params": list(self.get_encoder_params()),
            "initial_lr": self.base_lr,
        }
        if self.only_train_decoder:
            logger.info("Freezing encoder!")
            for param in encoder_params["params"]:
                param.requires_grad_(False)
        else:
            param_groups.append(encoder_params)

        decoder_params = {
            "name": "Decoder",
            "params": list(self.get_decoder_params()),
            "initial_lr": self.base_lr,
        }
        param_groups.append(decoder_params)

        # set up for autoencoder
        opt_ae = self.optimizer(param_groups)
        sched_ae = self.scheduler(opt_ae)
        opt_sched_ae = {
            "optimizer": opt_ae,
            "lr_scheduler": {"name": "lr-Autoencoder", "scheduler": sched_ae, "interval": "step"},
        }

        # set up for discriminator, if applicable
        disc_params = {
            "name": "Model",
            "params": list(self.get_discriminator_params()),
            "initial_lr": self.base_lr,
        }
        if len(disc_params["params"]) > 0 and self.loss.disc_factor > 0.0:
            opt_disc = self.optimizer([disc_params])
            sched_disc = self.scheduler(opt_disc)
        else:
            opt_disc, sched_disc = None, None
        opt_sched_disc = {
            "optimizer": opt_disc,
            "lr_scheduler": {"name": "lr-Discriminator", "scheduler": sched_disc, "interval": "step"},
        }

        if opt_disc is not None:
            return [opt_sched_ae, opt_sched_disc]
        return opt_sched_ae
