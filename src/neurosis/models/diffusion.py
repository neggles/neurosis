from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as L
import numpy as np
import torch
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import ListConfig
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.models.autoencoder import AutoencodingEngine
from neurosis.modules.diffusion import (
    BaseDiffusionSampler,
    Denoiser,
    StandardDiffusionLoss,
    UNetModel,
)
from neurosis.modules.diffusion.wrappers import OpenAIWrapper
from neurosis.modules.ema import LitEma
from neurosis.modules.encoders import GeneralConditioner
from neurosis.modules.encoders.embedding import AbstractEmbModel
from neurosis.utils import disabled_train, get_obj_from_str, log_txt_as_img, np_text_decode


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
        loss_fn: Optional[StandardDiffusionLoss],
        network_wrapper: Optional[str] = None,
        ckpt_path: Optional[PathLike] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast: bool = False,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
    ):
        super().__init__()

        self.log_keys = log_keys
        self.input_key = input_key

        network_wrapper = get_obj_from_str(network_wrapper) if network_wrapper is not None else OpenAIWrapper
        self.model = network_wrapper(model, compile_model=compile_model)
        self.denoiser = denoiser
        self.sampler = sampler
        self.conditioner = conditioner

        self.optimizer = optimizer
        self.scheduler = scheduler

        # do first stage model setup
        self._init_first_stage(first_stage_model)

        self.loss_fn = loss_fn

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        else:
            self.model_ema = None

        self.scale_factor: float = scale_factor
        self.first_stage_autocast: bool = not disable_first_stage_autocast
        self.no_cond_log: bool = no_cond_log

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
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.config.update(self.hparams)

    def init_from_ckpt(self, path: Path) -> None:
        if path.suffix == ".safetensors":
            sd = load_safetensors(path)
        elif path.suffix in CHECKPOINT_EXTNS:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        else:
            raise NotImplementedError(f"Unknown checkpoint extension {path.suffix}")

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def _init_first_stage(self, model: AutoencodingEngine):
        model.eval()
        model.freeze()
        model.train = disabled_train
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return batch[self.input_key]

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", enabled=self.first_stage_autocast):
            out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", enabled=self.first_stage_autocast):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def forward(self, x, batch) -> tuple[Tensor, dict[str, Tensor]]:
        loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"train/loss": loss_mean}
        return loss_mean, loss_dict

    def shared_step(self, batch: dict) -> Any:
        x = self.get_input(batch)
        x = self.encode_first_stage(x)
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx: int):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.scheduler is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("train/lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

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
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def configure_optimizers(self):
        network_params = list(self.model.parameters())
        embedder: AbstractEmbModel
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                network_params.extend(list(embedder.parameters()))

        optimizer = self.optimizer(network_params)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], scheduler
        else:
            return optimizer

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)

        def denoiser(input, sigma, c):
            return self.denoiser(self.model, input, sigma, c, **kwargs)

        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[2:]
        log = dict()

        embedder: AbstractEmbModel
        for embedder in self.conditioner.embedders:
            if ((self.log_keys is None) or (embedder.input_key in self.log_keys)) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = ["x".join([str(xx) for xx in x[i].tolist()]) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], np.bytes_):
                        x = np_text_decode(x)
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x: Tensor = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys if len(self.conditioner.embedders) > 0 else [],
        )

        sampling_kwargs = {}

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z: Tensor = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        log.update(self.log_conditionings(batch, N))

        for k in c:
            if isinstance(c[k], Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(self.device), (c, uc))

        if sample:
            with self.ema_scope("Plotting"):
                samples = self.sample(c, shape=z.shape[1:], uc=uc, batch_size=N, **sampling_kwargs)
            samples = self.decode_first_stage(samples)
            log["samples"] = samples
        return log
