import logging
import math
import re
from abc import abstractmethod
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning as L
import torch
from einops import rearrange
from packaging import version
from safetensors.torch import load_file as load_safetensors
from torch import Tensor, nn
from torch.optim import Optimizer

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.modules.autoencoding import AbstractRegularizer
from neurosis.modules.diffusion import Decoder, Encoder
from neurosis.modules.ema import LitEma
from neurosis.modules.regularizers import DiagonalGaussianRegularizer
from neurosis.utils import get_nested_attribute, get_obj_from_str

logger = logging.getLogger(__name__)


class AbstractAutoencoder(L.LightningModule):
    """
    This is the base class for all autoencoders, including image autoencoders, image autoencoders with discriminators,
    unCLIP models, etc. Hence, it is fairly general, and specific features
    (e.g. discriminator training, encoding, decoding) must be implemented in subclasses.
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,
        monitor: Union[None, str] = None,
        input_key: str = "jpg",
        ckpt_path: Union[None, str] = None,
        ignore_keys: Union[tuple, list] = tuple(),
    ):
        super().__init__()

        self.input_key = input_key
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

    @property
    def accumulate_grad_batches(self) -> int:
        try:
            return self.trainer.accumulate_grad_batches
        except Exception:
            return 1

    def init_from_ckpt(self, path: Path, ignore_keys: Union[tuple, list] = tuple()) -> None:
        path = Path(path)
        if path.suffix == ".safetensors":
            sd = load_safetensors(path)
        elif path.suffix in CHECKPOINT_EXTNS:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise NotImplementedError(f"Unknown checkpoint extension {path.suffix}")

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

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError()

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
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @abstractmethod
    def encode(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("encode()-method of abstract base class called")

    @abstractmethod
    def decode(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("decode()-method of abstract base class called")

    def instantiate_optimizer_from_config(self, params, lr: float, cfg: dict) -> Optimizer:
        logger.info(f"loading >>> {cfg['target']} <<< optimizer from config")

        opt_class = get_obj_from_str(cfg["target"])
        return opt_class(params, lr=lr, **cfg.get("init_args", dict()))

    @abstractmethod
    def configure_optimizers(self) -> Any:
        raise NotImplementedError("configure_optimizers()-method of abstract base class called")


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
        trainable_ae_params: Optional[list[list[str]]] = None,
        ae_optimizer_args: Optional[list[dict]] = None,
        trainable_disc_params: Optional[list[list[str]]] = None,
        disc_optimizer_args: Optional[list[dict]] = None,
        disc_start_iter: int = 0,
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
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor

        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = (
                ae_optimizer_args or [{} for _ in range(len(self.trainable_ae_params))],
            )

            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # makes type consitent

        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            self.disc_optimizer_args = (
                disc_optimizer_args or [{} for _ in range(len(self.trainable_disc_params))],
            )
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # makes type consitent

        self.additional_decode_keys = set(additional_decode_keys or [])

    def get_input(self, batch: Dict) -> Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first format (e.g., bchw instead if bhwc)
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = []
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            params += list(self.loss.get_trainable_autoencoder_parameters())
        if hasattr(self.regularization, "get_trainable_parameters"):
            params += list(self.regularization.get_trainable_parameters())
        params = params + list(self.encoder.parameters())
        params = params + list(self.decoder.parameters())
        return params

    def get_discriminator_params(self) -> list:
        if hasattr(self.loss, "get_trainable_parameters"):
            params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        else:
            params = []
        return params

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
        dec = self.decode(z, **additional_decode_kwargs)
        return z, dec, reg_log

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
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                "loss",
                aeloss.mean().detach(),
                prog_bar=True,
                logger=False,
                on_epoch=False,
                on_step=True,
            )
            return aeloss
        elif optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # -> discriminator always needs to return a tuple
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")

    def training_step(self, batch: dict, batch_idx: int):
        opts = self.optimizers()
        if not isinstance(opts, list):
            # Non-adversarial case
            opts = [opts]
        optimizer_idx = batch_idx % len(opts)
        if self.global_step < self.disc_start_iter:
            optimizer_idx = 0
        opt = opts[optimizer_idx]
        opt.zero_grad()
        with opt.toggle_model():
            loss = self.inner_training_step(batch, batch_idx, optimizer_idx=optimizer_idx)
            self.manual_backward(loss)

        opt.step()

    def validation_step(self, batch, batch_idx) -> Dict:
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
    ) -> tuple[list[Dict[str, Any]], int]:
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
        if self.trainable_ae_params is None:
            ae_params = self.get_autoencoder_params()
        else:
            ae_params, num_ae_params = self.get_param_groups(self.trainable_ae_params, self.ae_optimizer_args)
            logger.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        if self.trainable_disc_params is None:
            disc_params = self.get_discriminator_params()
        else:
            disc_params, num_disc_params = self.get_param_groups(
                self.trainable_disc_params, self.disc_optimizer_args
            )
            logger.info(f"Number of trainable discriminator parameters: {num_disc_params:,}")
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            (self.lr_g_factor or 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opts = [opt_ae]
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(
                disc_params, self.learning_rate, self.optimizer_config
            )
            opts.append(opt_disc)

        return opts

    @torch.no_grad()
    def log_images(self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs) -> dict:
        additional_decode_kwargs = {}
        x = self.get_input(batch)
        additional_decode_kwargs.update(
            {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
        )

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


class AutoencodingEngineLegacy(AutoencodingEngine):
    def __init__(
        self,
        *,
        embed_dim: int,
        loss: nn.Module = None,
        regularizer: Optional[AbstractRegularizer] = None,
        ddconfig: dict,
        **kwargs,
    ):
        self.max_batch_size = kwargs.pop("max_batch_size", None)

        if regularizer is None:
            regularizer = DiagonalGaussianRegularizer(sample=False)

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", tuple())
        super().__init__(
            encoder=Encoder(**ddconfig),
            decoder=Decoder(**ddconfig),
            regularizer=regularizer,
            loss=loss,
            **kwargs,
        )

        z_channels = ddconfig.get("z_channels", embed_dim)
        double_z = ddconfig.get("double_z", False)

        pq_in_channels = z_channels * 2 if double_z else z_channels
        pq_out_channels = embed_dim * 2 if double_z else embed_dim

        self.quant_conv = nn.Conv2d(
            in_channels=pq_in_channels,
            out_channels=pq_out_channels,
            kernel_size=1,
            bias=True,
        )
        self.post_quant_conv = nn.Conv2d(
            in_channels=embed_dim, out_channels=z_channels, kernel_size=1, bias=True
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def get_autoencoder_params(self) -> list:
        params = super().get_autoencoder_params()
        return params

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
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)

        return dec


class AutoencoderKL(AutoencodingEngineLegacy):
    """just an always-diagonal-gaussian-regulized autoencoder aka VAE"""

    def __init__(self, **kwargs):
        _ = kwargs.pop("regularizer", None)
        regularizer = DiagonalGaussianRegularizer(sample=False)
        super().__init__(regularizer=regularizer, **kwargs)


class AutoencoderKLInferenceWrapper(AutoencoderKL):
    def encode(self, x) -> Tensor:
        return super().encode(x).sample()


class IdentityFirstStage(AbstractAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_input(self, x: Any) -> Any:
        return x

    def encode(self, x: Any, *args, **kwargs) -> Any:
        return x

    def decode(self, x: Any, *args, **kwargs) -> Any:
        return x


class AEIntegerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        shape: Optional[tuple[int, int] | list[int]] = (16, 16),
        regularization_key: str = "regularization",
        encoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model

        if not hasattr(model, "encode") or not hasattr(model, "decode"):
            raise NotImplementedError("Need AE interface (encode and decode methods)")

        self.regularization = get_nested_attribute(model, regularization_key)
        self.shape = shape
        self.encoder_kwargs = encoder_kwargs or {"return_reg_log": True}

    def encode(self, x: Tensor) -> Tensor:
        if self.training:
            raise NotImplementedError(f"{self.__class__.__name__} only supports inference currently")

        _, log = self.model.encode(x, **self.encoder_kwargs)
        assert isinstance(log, dict)
        inds = log["min_encoding_indices"]
        return rearrange(inds, "b ... -> b (...)")

    def decode(self, inds: Tensor, shape: Optional[tuple | list] = None) -> Tensor:
        # expect inds shape (b, s) with s = h*w
        shape = shape or self.shape
        if shape is not None:
            if len(shape) != 2:
                raise NotImplementedError(f"Unhandled shape: {shape}")
            inds = rearrange(inds, "b (h w) -> b h w", h=shape[0], w=shape[1])
        h = self.regularization.get_codebook_entry(inds)  # (b, h, w, c)
        h = rearrange(h, "b h w c -> b c h w")
        return self.model.decode(h)
