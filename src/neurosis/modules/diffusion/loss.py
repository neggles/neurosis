import logging
import random
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from neurosis.modules.losses.perceptual import LPIPS
from neurosis.utils import append_dims

from ..encoders import GeneralConditioner
from .denoiser import Denoiser
from .denoiser_weighting import DenoiserWeighting
from .sampling.sigma_sampling import SigmaSampler

logger = logging.getLogger(__name__)


class DiffusionLoss(ABC, nn.Module):
    def __init__(
        self,
        noise_offset: float = 0.0,
        noise_offset_chance: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.noise_offset = min(max(noise_offset, 0.0), 1.0)
        self.noise_offset_chance = min(max(noise_offset_chance, 0.0), 1.0)

    @abstractmethod
    def get_loss(self, outputs: Tensor, target: Tensor, w: Tensor):
        raise NotImplementedError("Abstract base class was called ;_;")

    def apply_noise_offset(self, noise: Tensor, inputs: Tensor) -> Tensor:
        if self.noise_offset <= 0:
            return noise

        if self.noise_offset_chance == 1.0 or random.random() < self.noise_offset_chance:
            offset = torch.randn(inputs.shape[:2] + (1,) * (inputs.ndim - 2)).to(noise)
            return noise + self.noise_offset * offset

        return noise

    def get_noised_input(self, sigmas_bc: Tensor, noise: Tensor, inputs: Tensor) -> Tensor:
        noised_inputs = inputs + noise * sigmas_bc
        return noised_inputs

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        inputs: Tensor,
        batch: dict,
    ) -> Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, inputs, batch)

    @abstractmethod
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: dict,
        inputs: Tensor,
        batch: dict[str, Tensor],
    ) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")


class StandardDiffusionLoss(DiffusionLoss):
    def __init__(
        self,
        sigma_sampler: SigmaSampler,
        loss_weighting: DenoiserWeighting,
        loss_type: str = "l2",
        noise_offset: float = 0.0,
        noise_offset_chance: float = 0.0,
        input_keys: str | list[str] = [],
    ):
        super().__init__(noise_offset, noise_offset_chance)
        self.sigma_sampler = sigma_sampler
        self.loss_weighting = loss_weighting

        match loss_type.lower():
            case "l1":
                logger.debug("Using L1 loss")
                self.loss_type = "l1"
            case "l2" | "mse":
                logger.debug("Using L2 loss")
                self.loss_type = "l2"
            case "lpips":
                logger.debug("Using LPIPS perceptual loss")
                self.loss_type = "lpips"
                self.lpips = LPIPS().eval()
            case _:
                raise ValueError(f"Unknown loss type: '{loss_type}'")

        if not isinstance(input_keys, list):
            input_keys = [input_keys]
        self.input_keys = set(input_keys)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: dict,
        inputs: Tensor,
        batch: dict[str, Tensor],
    ) -> Tensor:
        # get extra inputs keys
        extra_inputs = {k: batch[k] for k in batch if k in self.input_keys}
        # get sigmas
        sigmas = self.sigma_sampler(inputs.shape[0]).to(inputs)

        # get noise
        noise = torch.randn_like(inputs)
        # apply offset
        noise = self.apply_noise_offset(noise, inputs)

        # expand sigmas to broadcast over batch
        sigmas_bc = append_dims(sigmas, inputs.ndim)
        # get noised input
        noised_input = self.get_noised_input(sigmas_bc, noise, inputs)

        # get model output
        outputs = denoiser(network, noised_input, sigmas, cond, **extra_inputs)

        # get loss weighting, expand to broadcast over batch
        weight = self.loss_weighting(sigmas)
        weight_bc = append_dims(weight, inputs.ndim)

        # get loss and return
        loss = self.get_loss(outputs, inputs, weight_bc)
        return loss.mean()

    def get_loss(self, outputs: Tensor, target: Tensor, weight: Tensor):
        if self.loss_type == "l2":
            return torch.mean((weight * (outputs - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            return torch.mean((weight * (outputs - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            loss = self.lpips(outputs, target).reshape(-1)
            return loss
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")
