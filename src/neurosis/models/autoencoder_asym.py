import logging
from contextlib import contextmanager
from functools import wraps
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
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Generator, Tensor, nn
from torch.optim.optimizer import Optimizer

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.modules.autoencoding.asymmetric import AsymmetricAutoencoderKL
from neurosis.modules.autoencoding.losses.discriminator_loss import GeneralLPIPSWithDiscriminator
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

    def get_loss(self, recons: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        if not isinstance(self.loss, str):
            warn(f"loss type {self.loss} is not a string, assuming it's a loss function")
            return self.loss(recons, target, *args, **kwargs)

        match self.loss:
            case "l1":
                return F.l1_loss(recons, target, reduction="mean")
            case "l2" | "mse":
                return F.mse_loss(recons, target, reduction="mean")
            case "nll":
                return F.nll_loss(recons, target, reduction="mean")
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
            loss = [self.get_loss(rec, inp) for inp, rec in zip(inputs, recons)]

        log_dict = {
            "inputs": inputs,
            "recons": recons,
            f"loss_{self.loss}": loss,
        }
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
        loss: GeneralLPIPSWithDiscriminator,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        accumulate_grad_batches: int = 1,
        **kwargs,
    ):
        super().__init__(loss="mse", ignore_keys=["loss"], *args, **kwargs)
        self.automatic_optimization = False

        self.loss: GeneralLPIPSWithDiscriminator = loss

        self.disc_start_iter = disc_start_iter
        self.diff_boost_factor = diff_boost_factor
        self.acc_grad_batches = accumulate_grad_batches

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

        if optimizer_idx == 0:
            # autoencoder loss
            ae_loss, log_dict_ae = self.loss(
                inputs=x,
                reconstructions=recons,
                optimizer_idx=optimizer_idx,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.loss_ema.update(log_dict_ae["loss/rec"].cpu().item())
            log_dict_ae.update({"train/loss/rec_ema": self.loss_ema.value})

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, sync_dist=False)
            loss = ae_loss
        elif optimizer_idx == 1:
            # discriminator
            disc_loss, log_dict_disc = self.loss(
                inputs=x,
                reconstructions=recons,
                optimizer_idx=optimizer_idx,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            # -> discriminator always needs to return a tuple
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True)
            loss = disc_loss
        else:
            raise ValueError(f"Unknown optimizer ID {optimizer_idx}")

        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int):
        opt_ae, opt_disc = self.optimizers()
        sched_ae, sched_disc = self.lr_schedulers()

        with opt_ae.toggle_model():
            loss = self.inner_training_step(batch, batch_idx, 0) / self.acc_grad_batches
            self.manual_backward(loss)

        if self.global_step > self.disc_start_iter:
            with opt_disc.toggle_model():
                loss = self.inner_training_step(batch, batch_idx, 1) / self.acc_grad_batches
                self.manual_backward(loss)

        if (batch_idx + 1) % self.acc_grad_batches == 0:
            opt_ae.step()
            sched_ae.step()
            opt_ae.zero_grad()
            if self.global_step > self.disc_start_iter:
                opt_disc.step()
                sched_disc.step()
                opt_disc.zero_grad()

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

        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": reg_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, recons, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, recons, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

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

        optimizer = self.optimizer(param_groups)
        opts = [optimizer]

        disc_params = {
            "name": "Decoder",
            "params": list(self.get_discriminator_params())
            + list(self.loss.get_trainable_autoencoder_parameters()),
        }
        if len(disc_params["params"]) > 0:
            opt_disc = self.optimizer([disc_params])
            opts.append(opt_disc)

        if self.base_lr is not None:
            for param_group in param_groups:
                param_group["initial_lr"] = self.base_lr

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            scheds = [scheduler]
            names = ["vae"]

            if len(disc_params["params"]) > 0:
                sched_disc = self.scheduler(opt_disc)
                scheds.append(sched_disc)
                names.append("disc")

        if len(opts) > 1:
            return [
                {
                    "optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step", "name": name},
                }
                for opt, sched, name in zip(opts, scheds, names)
            ]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "vae"},
            }
