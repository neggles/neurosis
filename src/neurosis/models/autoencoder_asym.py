import logging
from contextlib import contextmanager
from functools import wraps
from itertools import chain
from os import PathLike
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from lightning import pytorch as L
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Generator, Tensor, nn

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.modules.autoencoding.asymmetric import AsymmetricAutoencoderKL
from neurosis.modules.ema import LitEma
from neurosis.trainer.util import EMATracker

logger = logging.getLogger(__name__)


class AsymmetricAutoencodingEngine(L.LightningModule):
    def __init__(
        self,
        model: str | PathLike | AutoencoderKL | AsymmetricAutoencoderKL,
        optimizer: OptimizerCallable = ...,
        scheduler: LRSchedulerCallable = ...,
        loss_type: str = "mse",
        input_key: str = "image",
        log_keys: Optional[list] = None,
        asymmetric: bool = False,
        only_train_decoder: bool = False,
        base_lr: Optional[float] = None,
        ema_decay: Optional[float] = None,
        loss_ema_alpha: float = 0.02,
        **model_kwargs,
    ):
        super().__init__()
        logger.info(f"Initializing {self.__class__.__name__}")

        self.input_key = input_key
        self.base_lr = base_lr
        self.use_ema = ema_decay is not None

        self.log_keys = log_keys

        self.only_train_decoder = only_train_decoder
        self.asymmetric = asymmetric
        self.loss_ema = EMATracker(alpha=loss_ema_alpha)

        self.loss_type = loss_type

        self.optimizer = optimizer
        self.scheduler = scheduler

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))} weights.")

        if isinstance(model, (str, PathLike)):
            self.vae = load_vae_ckpt(Path(model), asymmetric=asymmetric, **model_kwargs)
        else:
            self.vae = model

        self.vae.disable_tiling()
        self.vae.disable_slicing()

        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler"])
        if self.only_train_decoder:
            self.freeze(encoder=True, decoder=False)

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

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

    def get_loss(self, model_output: Tensor, target: Tensor) -> Tensor:
        match self.loss_type:
            case "l1":
                return F.l1_loss(model_output, target, reduction="none").reshape(target.shape[0], -1).mean(1)
            case "l2" | "mse":
                return F.mse_loss(model_output, target, reduction="none").reshape(target.shape[0], -1).mean(1)
            case "cosine":
                return F.cosine_similarity(model_output.flatten(1), target.flatten(1), dim=1)
            case _:
                raise ValueError(f"loss type {self.loss_type} not supported")

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        x = self.get_input(batch)
        recon = self.forward(x)
        loss = self.get_loss(x, recon.sample)

        self.loss_ema.update(loss.mean().item())

        self.log_dict(
            {"train/loss": loss.mean(), "train/loss_ema": self.loss_ema.value},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=x.shape[0],
        )
        return loss.mean()

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
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer

    @torch.no_grad()
    def log_images(
        self,
        batch: dict,
        num_img: int = 1,
        sample: bool = True,
        ucg_keys: list[str] = None,
        **kwargs,
    ) -> dict:
        log_dict = {}

        x: Tensor = self.get_input(batch)[:num_img]

        with torch.inference_mode():
            recon = self.forward(x).sample

        log_dict["inputs"] = x
        log_dict["reconstructions"] = recon

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
