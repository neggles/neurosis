from typing import Optional, Union

import torch
from torch import Tensor, nn

from neurosis.modules.losses.lpips import LPIPS
from neurosis.utils import append_dims

from ..encoders import GeneralConditioner
from .denoiser import Denoiser
from .denoiser_weighting import DenoiserWeighting
from .sigma_sampling import DiffusionSampler


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler: DiffusionSampler,
        loss_weighting: DenoiserWeighting,
        loss_type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, list[str]]] = None,
    ):
        super().__init__()
        if loss_type not in ["l2", "l1", "lpips"]:
            raise ValueError(f"Unknown loss type {loss_type}, must be one of ['l2', 'l1', 'lpips']")

        self.sigma_sampler = sigma_sampler
        self.loss_weighting = loss_weighting

        self.loss_type = loss_type
        self.offset_noise_level = offset_noise_level

        if loss_type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def get_noised_input(self, sigmas_bc: Tensor, noise: Tensor, input: Tensor) -> Tensor:
        noised_input = input + noise * sigmas_bc
        return noised_input

    def forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        conditioner: GeneralConditioner,
        input: Tensor,
        batch: dict,
    ) -> Tensor:
        cond = conditioner(batch)
        return self._forward(network, denoiser, cond, input, batch)

    def _forward(
        self,
        network: nn.Module,
        denoiser: Denoiser,
        cond: dict,
        input: Tensor,
        batch: dict,
    ) -> tuple[Tensor, dict]:
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}
        sigmas = self.sigma_sampler(input.shape[0]).to(input)

        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            offset_shape = (input.shape[0], input.shape[1])
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(offset_shape, device=input.device),
                input.ndim,
            )
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = self.get_noised_input(sigmas_bc, noise, input)

        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(self.loss_weighting(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output: Tensor, target: Tensor, w: Tensor):
        if self.loss_type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")
