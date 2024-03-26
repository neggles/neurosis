import logging
import random
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from neurosis.modules.losses.functions import BatchL1Loss, BatchMSELoss
from neurosis.utils import append_dims

from ..encoders import GeneralConditioner
from .denoiser import Denoiser
from .denoiser_weighting import DenoiserWeighting, VWeighting
from .sampling.sigma_generators import SigmaGenerator

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
        sigma_generator: SigmaGenerator,
        loss_weighting: DenoiserWeighting,
        loss_type: str = "l2",
        snr_gamma: float = 0.0,
        noise_offset: float = 0.0,
        noise_offset_chance: float = 0.0,
        input_keys: str | list[str] = [],
    ):
        super().__init__(noise_offset, noise_offset_chance)
        self.sigma_generator = sigma_generator
        self.loss_weighting = loss_weighting
        self.snr_gamma = snr_gamma
        self.v_prediction = issubclass(self.loss_weighting.__class__, VWeighting)

        match loss_type.lower():
            case "l1":
                logger.debug("Using L1 loss")
                self.loss_type = "l1"
                self.loss = BatchL1Loss(reduction="mean")
            case "l2" | "mse":
                logger.debug("Using L2 loss")
                self.loss_type = "l2"
                self.loss = BatchMSELoss(reduction="mean")
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
        sigmas = self.sigma_generator(inputs.shape[0]).to(inputs)

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

        # get loss weighting
        weight = self.loss_weighting(sigmas)

        # get loss
        loss = self.get_loss(outputs, inputs, weight)

        # if we're doing min-snr weighting, do that
        if self.snr_gamma > 0:
            snr = 1 / sigmas**2
            min_snr_gamma = torch.min(snr, torch.full_like(snr, self.snr_gamma))
            if self.v_prediction:
                snr_weight = min_snr_gamma.div(snr + 1)
            else:
                snr_weight = min_snr_gamma.div(snr)
            loss *= snr_weight

        return loss

    def get_loss(self, outputs: Tensor, target: Tensor, weight: Tensor):
        if self.loss_type in ["l1", "l2"]:
            return self.loss(outputs.float(), target.float()) * weight.float()
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")
