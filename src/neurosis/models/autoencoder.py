import logging
import math
import re
from abc import abstractmethod
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Union

import lightning as L
import torch
from einops import rearrange
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from packaging import version
from safetensors.torch import load_file as load_safetensors
from torch import Tensor, nn

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.modules.autoencoding import AbstractRegularizer
from neurosis.modules.diffusion import Decoder, Encoder
from neurosis.modules.ema import LitEma
from neurosis.modules.regularizers import DiagonalGaussianRegularizer
from neurosis.utils import get_nested_attribute

logger = logging.getLogger(__name__)


class AbstractAutoencoder(L.LightningModule):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Optional[float] = None,
        monitor: Optional[str] = None,
        input_key: str = "jpg",
        ckpt_path: Optional[str] = None,
        ignore_keys: tuple | list = tuple(),
        base_lr: Optional[float] = None,
    ):
        super().__init__()
        self.encoder: Encoder
        self.decoder: Decoder

        self.input_key = input_key
        self.base_lr = base_lr
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))} weights.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if version.parse(L.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    def init_from_ckpt(self, path: Path, ignore_keys: Union[tuple, list] = tuple()) -> None:
        path = Path(path)
        if path.suffix == ".safetensors":
            sd = load_safetensors(path)
        elif path.suffix in CHECKPOINT_EXTNS:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise ValueError(f"Unknown checkpoint extension {path.suffix}")

        keys = list(sd.keys())

        if self.use_ema is False:
            if ignore_keys is None:
                ignore_keys = ("model_ema", "model_ema.ema_decay", "model_ema.ema_buffer")
            else:
                ignore_keys = tuple(ignore_keys) + (
                    "model_ema",
                    "model_ema.ema_decay",
                    "model_ema.ema_buffer",
                )

        for k in keys:
            for ik in ignore_keys:
                if re.search(ik, k):
                    logger.info(f"Deleting key {k} from state_dict.")
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            logger.warn(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")

    def get_input(self, batch) -> Any:
        return batch[self.input_key]

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

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

    @abstractmethod
    def encode(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def decode(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def configure_optimizers(self) -> Any:
        raise NotImplementedError("Abstract base class was called ;_;")


class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    def __init__(
        self,
        *args,
        encoder: Encoder,
        decoder: Decoder,
        loss: nn.Module,
        regularizer: Optional[AbstractRegularizer] = None,
        optimizer_config: Optional[dict] = None,
        lr_g_factor: float = 1.0,
        optimizer: Optional[OptimizerCallable] = None,
        scheduler: Optional[LRSchedulerCallable] = None,
        disc_start: int = 0,
        diff_boost_factor: float = 3.0,
        additional_decode_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # pytorch lightning

        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.regularization = regularizer
        self.optimizer_config = optimizer_config

        self.diff_boost_factor = diff_boost_factor
        self.disc_start = disc_start
        self.lr_g_factor = lr_g_factor

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.additional_decode_keys = set(additional_decode_keys or [])

    def get_input(self, batch: dict) -> Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self, decoder_only: bool = False) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())

        params += list(self.decoder.parameters())
        if decoder_only:
            return params

        params += list(self.encoder.parameters())
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            return list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            return []

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def encode(
        self,
        x: Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
    ) -> Union[Tensor, tuple[Tensor, dict]]:
        z = self.encoder(x)
        if unregularized:
            return z, dict()
        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        x = self.decoder(z, **kwargs)
        return x

    def forward(self, x: Tensor, **additional_decode_kwargs) -> tuple[Tensor, Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)
        x = self.decode(z, **additional_decode_kwargs)
        return z, x, reg_log

    def inner_training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0) -> Tensor:
        x = self.get_input(batch)
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }
        z, xrec, regularization_log = self(x, **additional_decode_kwargs)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": optimizer_idx,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "train",
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()

        if optimizer_idx == 0:
            # autoencode
            out_loss = self.loss(x, xrec, **extra_info)
            if isinstance(out_loss, tuple):
                aeloss, log_dict_ae = out_loss
            else:
                # simple loss function
                aeloss = out_loss
                log_dict_ae = {"train/loss/rec": aeloss.detach()}

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True)
            return discloss
        else:
            raise ValueError(f"Unknown optimizer ID {optimizer_idx}")

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(batch, batch_idx, optimizer_idx)
            self.manual_backward(loss)
        opt.step()

    def validation_step(self, batch, batch_idx) -> dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        if hasattr(self.loss, "forward_keys"):
            extra_info = {
                "z": z,
                "optimizer_idx": 0,
                "global_step": self.global_step,
                "last_layer": self.get_last_layer(),
                "split": "val" + postfix,
                "regularization_log": regularization_log,
                "autoencoder": self,
            }
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            extra_info = dict()
        out_loss = self.loss(x, xrec, **extra_info)
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss
        else:
            # simple loss function
            aeloss = out_loss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}
        full_log_dict = log_dict_ae

        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            full_log_dict.update(log_dict_disc)
        self.log(
            f"val{postfix}/loss/rec",
            log_dict_ae[f"val{postfix}/loss/rec"],
            sync_dist=True,
        )
        self.log_dict(full_log_dict, sync_dist=True)
        return full_log_dict

    def get_param_groups(
        self, parameter_names: list[list[str]], optimizer_args: list[dict]
    ) -> tuple[list[dict[str, Any]], int]:
        groups = []
        num_params = 0
        for names, args in zip(parameter_names, optimizer_args):
            params = []
            for pattern_ in names:
                pattern_params = []
                pattern = re.compile(pattern_)
                for p_name, param in self.named_parameters():
                    if re.match(pattern, p_name):
                        pattern_params.append(param)
                        num_params += param.numel()
                if len(pattern_params) == 0:
                    logger.warn(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)
            groups.append({"params": params, **args})
        return groups, num_params

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        ae_params = self.get_autoencoder_params()
        opt_ae = self.optimizer(ae_params)
        opts = [opt_ae]

        disc_params = self.get_discriminator_params()
        if len(disc_params) > 0:
            opt_disc = self.optimizer(disc_params)
            opts.append(opt_disc)

        return opts

    @torch.no_grad()
    def log_images(
        self,
        batch: dict,
        num_img: int = 1,
        additional_log_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        x = self.get_input(batch)[:num_img]
        additional_decode_kwargs = {
            key: batch[key] for key in self.additional_decode_keys.intersection(batch)
        }

        _, xrec, _ = self(x, **additional_decode_kwargs)
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        diff.clamp_(0, 1.0)

        log_dict = {
            "inputs": x,
            "reconstructions": xrec,
            "diff": 2.0 * diff - 1.0,
            # diff_boost shows location of small errors, by boosting their brightness.
            "diff_boost": 2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1,
        }

        if hasattr(self.loss, "log_images"):
            log_dict.update(self.loss.log_images(x, xrec))

        with self.ema_scope():
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            diff_ema.clamp_(0, 1.0)
            log_dict.update(
                {
                    "reconstructions_ema": xrec_ema,
                    "diff_ema": 2.0 * diff_ema - 1.0,
                    "diff_boost_ema": 2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1,
                }
            )

        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            log_dict[log_str] = xrec_add

        return log_dict

    def unfreeze_decoder(self):
        if not hasattr(self, "decoder"):
            raise ValueError("No decoder found!")
        self.decoder.train()
        self.decoder.requires_grad_(True)


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(
        self,
        *,
        embed_dim: int,
        loss: Optional[nn.Module] = None,
        regularizer: Optional[AbstractRegularizer] = None,
        ddconfig: dict = {},
        **kwargs,
    ):
        self.max_batch_size = kwargs.pop("max_batch_size", None)

        if regularizer is None:
            regularizer = DiagonalGaussianRegularizer(sample=False)

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", tuple())

        ddconfig["embed_dim"] = embed_dim
        super().__init__(
            encoder=Encoder(**ddconfig),
            decoder=Decoder(**ddconfig),
            regularizer=regularizer,
            loss=loss or nn.Identity(),
            **kwargs,
        )

        quant_conv_in_ch = (1 + ddconfig["double_z"]) * ddconfig["z_channels"]
        quant_conv_out_ch = (1 + ddconfig["double_z"]) * embed_dim

        self.quant_conv = nn.Conv2d(quant_conv_in_ch, quant_conv_out_ch, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.encoder.max_batch_size = self.max_batch_size
        self.decoder.max_batch_size = self.max_batch_size

    def encode(self, x: Tensor, return_reg_log: bool = False) -> Union[Tensor, tuple[Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: Tensor, **decoder_kwargs) -> Tensor:
        if self.max_batch_size is None:
            x = self.post_quant_conv(z)
            x = self.decoder(x, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            x = list()
            for i_batch in range(n_batches):
                x_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                x_batch = self.decoder(x_batch, **decoder_kwargs)
                x.append(x_batch)
            x = torch.cat(x, 0)

        return x


class AutoencoderKL(AutoencodingEngineLegacy):
    """just an always-diagonal-gaussian-regulized autoencoder aka VAE"""

    @wraps(AutoencodingEngineLegacy.__init__)
    def __init__(
        self,
        *,
        regularizer: Optional[nn.Module] = None,
        train_decoder_only: bool = False,
        **kwargs,
    ):
        regularizer = DiagonalGaussianRegularizer(sample=False)
        super().__init__(regularizer=regularizer, **kwargs)
        if train_decoder_only:
            self.freeze()
            self.unfreeze_decoder()


class FSDPAutoencoderKL(AutoencodingEngineLegacy):
    @wraps(AutoencodingEngineLegacy.__init__)
    def __init__(
        self,
        *,
        embed_dim: int,
        regularizer: Optional[nn.Module] = None,
        ddconfig: dict = {},
        loss: nn.Module = nn.Identity(),
        standalone: Optional[Any] = None,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.max_batch_size = kwargs.pop("max_batch_size", None)

        if standalone is not None:
            # this keeps being False. I do not know why. it should not be anything.
            logger.warn(f"standalone is {standalone} somehow")

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", tuple())

        # set embed dim
        ddconfig["embed_dim"] = embed_dim
        # override standalone
        ddconfig["standalone"] = True
        # always gaussian, always
        regularizer = DiagonalGaussianRegularizer(sample=False)

        AutoencodingEngine.__init__(
            self,
            encoder=Encoder(**ddconfig),
            decoder=Decoder(**ddconfig),
            regularizer=regularizer,
            loss=loss,
            **kwargs,
        )
        self.encoder.max_batch_size = self.max_batch_size
        self.decoder.max_batch_size = self.max_batch_size

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_from_ckpt(self, path: Path, ignore_keys: Union[tuple, list] = tuple()) -> None:
        path = Path(path)
        if path.suffix == ".safetensors":
            sd = load_safetensors(path)
        elif path.suffix in CHECKPOINT_EXTNS:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise ValueError(f"Unknown checkpoint extension {path.suffix}")

        if self.use_ema is False:
            if ignore_keys is None:
                ignore_keys = tuple()
            ignore_keys = ignore_keys + ("model_ema", "model_ema.ema_decay", "model_ema.ema_buffer")

        for ik in ignore_keys:
            for k in list(sd.keys()):
                if re.search(re.escape(ik), k):
                    logger.info(f"Deleting key {k} from state_dict.")
                    _ = sd.pop(k, None)

        keymap = {
            "quant_conv": "encoder.quant_conv",
            "post_quant_conv": "decoder.post_quant_conv",
        }
        for sd_prefix, model_prefix in keymap.items():
            for pname in ("weight", "bias"):
                pkey, mkey = f"{sd_prefix}.{pname}", f"{model_prefix}.{pname}"
                if pkey in sd:
                    logger.info(f"Remapping {pkey} to {mkey}")
                    sd[mkey] = sd.pop(pkey)

        missing, unexpected = self.load_state_dict(sd, strict=False)
        logger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            logger.warn(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logger.info(f"Unexpected Keys: {unexpected}")

    def encode(self, x: Tensor, return_reg_log: bool = False) -> Union[Tensor, tuple[Tensor, dict]]:
        if self.max_batch_size is None:
            z = self.encoder(x)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: Tensor, **decoder_kwargs) -> Tensor:
        if self.max_batch_size is None:
            x = self.decoder(z, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            x = list()
            for i_batch in range(n_batches):
                x_batch = self.decoder(z[i_batch * bs : (i_batch + 1) * bs], **decoder_kwargs)
                x.append(x_batch)
            x = torch.cat(x, 0)

        return x


class AutoencoderKLInferenceWrapper(AutoencoderKL):
    def encode(self, x) -> Tensor:
        return super().encode(x).sample()


class IdentityFirstStage(AbstractAutoencoder):
    def __init__(self, input_key: str = "jpg", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_key = input_key
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def get_input(self, x: Any) -> Any:
        return x[self.input_key]

    def encode(self, x: Any, *_, **__) -> Any:
        return self.encoder(x)

    def decode(self, x: Any, *_, **__) -> Any:
        return self.decoder(x)


class AEIntegerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        shape: Optional[tuple[int, int] | list[int]] = (16, 16),
        regularization_key: str = "regularization",
        encoder_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model

        if not hasattr(model, "encode") or not hasattr(model, "decode"):
            raise ValueError("Need AE interface (encode and decode methods)")

        self.regularization = get_nested_attribute(model, regularization_key)
        self.shape = shape
        self.encoder_kwargs = encoder_kwargs or {"return_reg_log": True}

    def encode(self, x: Tensor) -> Tensor:
        if self.training:
            raise RuntimeError(f"{self.__class__.__name__} only supports inference currently")

        _, log = self.model.encode(x, **self.encoder_kwargs)
        if not isinstance(log, dict):
            raise ValueError(f"Log was not a dict: {log}")
        indices = log["min_encoding_indices"]
        return rearrange(indices, "b ... -> b (...)")

    def decode(self, indices: Tensor, shape: Optional[tuple | list] = None) -> Tensor:
        # expect indices shape (b, s) with s = h*w
        shape = shape or self.shape
        if shape is not None:
            if len(shape) != 2:
                raise ValueError(f"Invalid input shape: {shape}")
            indices = rearrange(indices, "b (h w) -> b h w", h=shape[0], w=shape[1])
        h = self.regularization.get_codebook_entry(indices)  # (b, h, w, c)
        h = rearrange(h, "b h w c -> b c h w")
        return self.model.decode(h)
