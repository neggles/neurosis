from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class DenoiserScaling(ABC):
    @abstractmethod
    def __call__(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError("Abstract base class was called ;_;")


class EDMScaling(DenoiserScaling):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise


class EpsScaling(DenoiserScaling):
    def __call__(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c_skip = torch.ones_like(sigma, device=sigma.device)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScaling(DenoiserScaling):
    def __call__(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise


class VScalingWithEDMcNoise(DenoiserScaling):
    def __call__(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise
