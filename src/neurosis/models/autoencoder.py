import re
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import lightning as L
import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from omegaconf import ListConfig
from packaging import version
from safetensors.torch import load_file as load_safetensors
from torch import Tensor, nn

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.modules.autoencoding import AbstractRegularizer
from neurosis.modules.diffusion import Decoder, Encoder
from neurosis.modules.distributions import DiagonalGaussianDistribution
from neurosis.modules.ema import LitEma


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
        ignore_keys: Union[Tuple, list, ListConfig] = tuple(),
    ):
        super().__init__()
        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

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

    def init_from_ckpt(self, path: Path, ignore_keys: Union[Tuple, list, ListConfig] = tuple()) -> None:
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
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    @abstractmethod
    def get_input(self, batch) -> Any:
        raise NotImplementedError()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
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
        optimizer: Union[Dict, None] = None,
        lr_g_factor: float = 1.0,
        trainable_ae_params: Optional[list[list[str]]] = None,
        ae_optimizer_args: Optional[list[dict]] = None,
        trainable_disc_params: Optional[list[list[str]]] = None,
        disc_optimizer_args: Optional[list[dict]] = None,
        disc_start_iter: int = 0,
        diff_boost_factor: float = 3.0,
        ckpt_engine: Union[None, str, dict] = None,
        ckpt_path: Optional[str] = None,
        additional_decode_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # todo: add options to freeze encoder/decoder

        self.automatic_optimization = False  # pytorch lightning

        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.regularization = regularizer
        self.optimizer = optimizer

        self.diff_boost_factor = diff_boost_factor
        self.disc_start_iter = disc_start_iter
        self.lr_g_factor = lr_g_factor
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            self.ae_optimizer_args = ae_optimizer_args or [{} for _ in range(len(self.trainable_ae_params))]
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

        self.apply_ckpt(ckpt_path, ckpt_engine)
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
    ) -> Union[Tensor, Tuple[Tensor, dict]]:
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

    def forward(self, x: Tensor, **additional_decode_kwargs) -> Tuple[Tensor, Tensor, dict]:
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

    def _validation_step(self, batch, batch_idx, postfix="") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        aeloss, log_dict_ae = self.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )
        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        log_dict_ae.update(log_dict_disc)
        self.log_dict(log_dict_ae)
        return log_dict_ae

    def configure_optimizers(self) -> Any:
        ae_params = self.get_autoencoder_params()
        disc_params = self.get_discriminator_params()

        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            (self.lr_g_factor or 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opt_disc = self.instantiate_optimizer_from_config(
            disc_params, self.learning_rate, self.optimizer_config
        )

        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch)
        _, xrec, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        with self.ema_scope():
            _, xrec_ema, _ = self(x)
            log["reconstructions_ema"] = xrec_ema
        return log


class AutoencoderKL(AutoencodingEngine):
    def __init__(
        self,
        *,
        embed_dim: int,
        z_channels: Optional[int] = None,
        loss: nn.Module = None,
        ddconfig: dict,
        **kwargs,
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", tuple())
        super().__init__(
            encoder=nn.Identity,
            decoder=nn.Identity,
            regularizer=nn.Identity,
            loss=loss,
            **kwargs,
        )

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        z_channels = z_channels if z_channels is not None else 2 * embed_dim

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x) -> DiagonalGaussianDistribution:
        if self.training:
            raise NotImplementedError(f"{self.__class__.__name__} does not support training yet.")
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, **decoder_kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, **decoder_kwargs)
        return dec


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
