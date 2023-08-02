from typing import List, Optional, Union

import torch
from omegaconf import ListConfig
from torch import Tensor, nn

from neurosis.modules.diffusion.sigma_sampling import DiffusionSampler
from neurosis.modules.losses.lpips import LPIPS
from neurosis.utils import append_dims


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler: DiffusionSampler,
        type: str = "l2",
        offset_noise_level: float = 0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()
        if type not in ["l2", "l1", "lpips"]:
            raise ValueError(f"Unknown loss type {type}, must be one of ['l2', 'l1', 'lpips']")

        self.sigma_sampler = sigma_sampler

        self.loss_type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network: nn.Module, denoiser, conditioner, input: Tensor, batch: Tensor) -> Tensor:
        cond = conditioner(batch)
        additional_model_inputs = {key: batch[key] for key in self.batch2model_keys.intersection(batch)}

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        noised_input = input + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(network, noised_input, sigmas, cond, **additional_model_inputs)
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output: Tensor, target: Tensor, w: Tensor) -> Tensor:
        if self.loss_type == "l2":
            return torch.mean((w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "l1":
            return torch.mean((w * (model_output - target).abs()).reshape(target.shape[0], -1), 1)
        elif self.loss_type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
