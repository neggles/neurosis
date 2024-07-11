import logging
import random
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from neurosis.modules.losses.functions import BatchL1Loss, BatchMSELoss
from neurosis.modules.losses.types import DiffusionObjective, GenericLoss
from neurosis.utils import append_dims

from ..encoders import GeneralConditioner
from .denoiser import Denoiser
from .denoiser_weighting import DenoiserWeighting
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

    def apply_noise_offset(self, noise: Tensor, inputs: Tensor) -> Tensor:
        if self.noise_offset <= 0:
            return noise

        if self.noise_offset_chance == 1.0 or random.random() < self.noise_offset_chance:
            offset = torch.randn(inputs.shape[:2] + (1,) * (inputs.ndim - 2)).to(noise)
            return noise + self.noise_offset * offset

        return noise

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        inputs: Tensor,
        batch: dict,
        return_dict: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, inputs, batch, return_dict)

    @abstractmethod
    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: dict,
        inputs: Tensor,
        batch: dict[str, Tensor],
        return_dict: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def get_loss(self, outputs: Tensor, target: Tensor, w: Tensor):
        raise NotImplementedError("Abstract base class was called ;_;")


class StandardDiffusionLoss(DiffusionLoss):
    def __init__(
        self,
        sigma_generator: SigmaGenerator,
        loss_weighting: DenoiserWeighting,
        loss_type: GenericLoss = GenericLoss.L2,
        snr_gamma: float = 0.0,
        noise_offset: float = 0.0,
        noise_offset_chance: float = 0.0,
        input_keys: str | list[str] = [],
        objective_type: DiffusionObjective = DiffusionObjective.EDM,
    ):
        super().__init__(noise_offset, noise_offset_chance)
        self.sigma_generator = sigma_generator
        self.loss_weighting = loss_weighting
        self.snr_gamma = snr_gamma
        self.objective_type = objective_type

        match loss_type.lower():
            case GenericLoss.L1:
                logger.debug("Using L1 loss")
                self.loss_type = "l1"
                self.loss = BatchL1Loss(reduction="mean")
            case GenericLoss.L2 | GenericLoss.MSE:
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
        return_dict: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        # get extra inputs keys
        extra_inputs = {k: batch[k] for k in batch if k in self.input_keys}
        # gen timestep
        t = torch.rand((inputs.shape[0],), dtype=torch.float64)
        # get sigmas
        sigmas = self.sigma_generator(inputs.shape[0], t).to(inputs)
        # get noise
        noise = torch.randn_like(inputs)
        # apply offset
        noise = self.apply_noise_offset(noise, inputs)
        # expand sigmas to broadcast over batch
        sigmas_bc = append_dims(sigmas, inputs.ndim)

        match self.objective_type:
            case "rf":
                # get latent state
                alpha = 1.0 - sigmas_bc
                z_t = alpha * inputs + sigmas_bc * noise
                # get eps output
                eps_output = denoiser(network, z_t, sigmas, cond, "F", **extra_inputs)
                # get weighting
                weight = self.loss_weighting(sigmas)
                # get loss
                loss = self.get_loss(eps_output, noise, weight)
            case "edm":
                # get latent state
                z_t = inputs + sigmas_bc * noise
                # get denoised output
                D_output = denoiser(network, z_t, sigmas, cond, "D", **extra_inputs)
                # get weighting
                weight = self.loss_weighting(sigmas)
                # get loss
                loss = self.get_loss(D_output, inputs, weight)
            case _:
                raise ValueError(f"Unknown objective type: '{self.objective_type}'")
        if return_dict:
            return loss, {"sigmas": sigmas, "t": t}
        return loss

    def get_loss(self, outputs: Tensor, target: Tensor, weight: Tensor):
        if self.loss_type in ["l1", "l2"]:
            return self.loss(outputs.float(), target.float()) * weight.float()
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")
