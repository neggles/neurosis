import logging
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.modules.losses.dreamsim.model import DreamsimBackbone, DreamsimEnsemble, DreamsimModel
from neurosis.modules.losses.functions import BatchL1Loss, BatchMSELoss
from neurosis.modules.losses.types import GenericLoss

logger = logging.getLogger(__name__)


class AutoencoderDreamsim(nn.Module):
    def __init__(
        self,
        dreamsim_name_or_path: PathLike = "neggles/dreamsim:ensemble_vitb16",
        dreamsim_weight: float = 1.0,
        dreamsim_ensemble: bool = True,
        recon_type: GenericLoss | str = GenericLoss.L1,
        recon_weight: float = 1.0,
        resize_input: bool = False,
        resize_target: bool = False,
        extra_log_keys: list[str] = [],
        rescale_input: bool = True,
        delay_setup: bool = False,
        dreamsim_compile: bool = False,
    ):
        super().__init__()
        # set up reconstruction loss
        self.recon_type = recon_type
        match self.recon_type:
            case GenericLoss.L1:
                self.recon_loss = BatchL1Loss(reduction="mean")
            case GenericLoss.L2 | GenericLoss.MSE:
                self.recon_loss = BatchMSELoss(reduction="mean")
            case _:
                raise ValueError(f"Unknown reconstruction loss type {self.recon_type}")
        self.recon_weight = recon_weight

        # validate the resize options (not likely to be used anyway)
        if resize_input and resize_target:
            raise ValueError("Only one of resize_input and resize_target can be True")
        self.resize_input_to_target = resize_input
        self.resize_target_to_input = resize_target
        self.rescale_input = rescale_input

        # set up perceptual loss
        self._dreamsim_cls = DreamsimEnsemble if dreamsim_ensemble else DreamsimModel
        self._dreamsim_path = Path(dreamsim_name_or_path)
        self._dreamsim_name = dreamsim_name_or_path
        self.dreamsim_loss: DreamsimBackbone = None
        self.dreamsim_weight = dreamsim_weight

        # keys to be used in the forward method
        self.forward_keys = ["split"]
        # extra log keys
        self.extra_log_keys = set(extra_log_keys)
        # whether to try TorchInductor compilation (fraught with peril)
        self.dreamsim_compile = dreamsim_compile

        if delay_setup:
            logger.info("Delaying Dreamsim model loading until configure_model() call")
        else:
            self.configure_model()

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def dtype(self):
        return self.parameters().__next__().dtype

    def configure_model(self) -> None:
        if self.dreamsim_loss is not None:
            # make sure we're not training the perceptual loss just in case
            self.dreamsim_loss = self.dreamsim_loss.requires_grad_(False)
            return

        if self._dreamsim_path.is_dir() and self._dreamsim_path.joinpath("config.json").exists():
            self.dreamsim_loss: DreamsimBackbone = self._dreamsim_cls.from_pretrained(
                self._dreamsim_path, local_files_only=True
            ).eval()
        else:
            if ":" in self._dreamsim_name:
                repo_id, revision = self._dreamsim_name.split(":")
            else:
                repo_id, revision = self._dreamsim_name, "main"
            logger.info(f"Loading Dreamsim from HuggingFace repo {repo_id} at revision '{revision}'")
            self.dreamsim_loss: DreamsimBackbone = self._dreamsim_cls.from_pretrained(
                repo_id, revision=revision
            ).eval()

        self.dreamsim_loss = self.dreamsim_loss.requires_grad_(False)
        if self.dreamsim_compile:
            self.dreamsim_loss = torch.compile(self.dreamsim_loss, dynamic=False, mode="reduce-overhead")

    def forward(
        self,
        inputs: Tensor,
        recons: Tensor,
        split: str = "train",
        **kwargs,
    ) -> tuple[Tensor, dict]:
        if self.resize_input_to_target:
            inputs = F.interpolate(inputs, recons.shape[2:], mode="bicubic", antialias=True)
        if self.resize_target_to_input:
            recons = F.interpolate(recons, inputs.shape[2:], mode="bicubic", antialias=True)

        # do reconstruction and perceptual loss
        inputs = inputs.clamp(-1.0, 1.0).contiguous()
        recons = recons.clamp(-1.0, 1.0).contiguous()

        rec_loss = self.recon_loss(inputs, recons)
        rec_loss = rec_loss * self.recon_weight

        if self.rescale_input:
            inputs = (inputs / 2 + 0.5).clamp(0.0, 1.0)
            recons = (recons / 2 + 0.5).clamp(0.0, 1.0)

        ds_inputs = torch.stack([inputs, recons], dim=0)
        ds_loss = self.dreamsim_loss(ds_inputs).sum().relu()
        ds_loss = ds_loss * self.dreamsim_weight

        loss = rec_loss + ds_loss

        log_loss = loss.detach().clone().mean()
        log_rec_loss = rec_loss.detach().mean()
        log_ds_loss = ds_loss.detach().mean()

        log_dict = {
            f"{split}/loss/total": log_loss,
            f"{split}/loss/rec": log_rec_loss,
            f"{split}/loss/p": log_ds_loss,
        }

        return loss, log_dict

    @torch.no_grad()
    def log_loss(
        self,
        inputs: Tensor,
        recons: Tensor,
        split: str = "train",
        **kwargs,
    ) -> dict[str, Tensor]:
        _, log_dict = self.forward(inputs=inputs, recons=recons, split=split, **kwargs)

        return log_dict
