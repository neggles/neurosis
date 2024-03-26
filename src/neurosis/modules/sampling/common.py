from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


class DiffusionSampler2(ABC, nn.Module):
    sigmas: Tensor
    log_sigmas: Tensor
    sigma_data: Optional[float | Tensor]

    def set_sigmas(self, sigmas: Tensor, sigma_data: Optional[float | Tensor] = None):
        self.sigma_data = sigma_data
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("log_sigmas", sigmas.log())

    @property
    def sigma_min(self) -> Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> Tensor:
        return self.sigmas[-1]

    @abstractmethod
    def timestep(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def sigma(self, timestep: int | float | Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent
        return self.sigma(torch.tensor(percent * 999.0)).item()


class SigmaScheduler(ABC):
    def __init__(self, sampler: DiffusionSampler2):
        super().__init__()
        self.sampler = sampler

    def __call__(self, n_steps: int, dtype: torch.dtype = torch.float32) -> Tensor:
        return self.get_schedule(n_steps, dtype)

    @abstractmethod
    def get_schedule(self, n_steps: int, dtype: torch.dtype = torch.float32) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")


class NoiseScaling(ABC, nn.Module):
    def __init__(self, sigma_data: float = 1.0):
        super().__init__()
        self.sigma_data = sigma_data

    @abstractmethod
    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def calculate_denoised(self, sigma: Tensor, model_output: Tensor, model_input: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def noise_scaling(self, sigma: Tensor, noise: Tensor, latents: Tensor, max_denoise: bool = False):
        raise NotImplementedError("Abstract base class was called ;_;")
