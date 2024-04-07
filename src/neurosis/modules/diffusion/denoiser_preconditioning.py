from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class DenoiserPreconditioning(ABC):
    def __call__(self, sigma: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.get_c_skip(sigma), self.get_c_out(sigma), self.get_c_in(sigma), self.get_c_noise(sigma)

    @abstractmethod
    def get_c_skip(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def get_c_out(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def get_c_in(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def get_c_noise(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")

    def get_snr(self, sigma: Tensor) -> Tensor:
        # this assumes unit variance for the data
        return 1 / sigma**2.0


class EpsPreconditioning(DenoiserPreconditioning):
    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return torch.ones_like(sigma, device=sigma.device)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return -sigma

    def get_c_in(self, sigma: Tensor) -> Tensor:
        return 1.0 / (sigma**2.0 + 1.0) ** 0.5

    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return sigma.clone()


class VPreconditioning(EpsPreconditioning):
    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return 1.0 / (sigma**2 + 1.0)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return -sigma / (sigma**2 + 1.0) ** 0.5


class VPreconditioningWithEDMcNoise(VPreconditioning):
    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return 0.25 * sigma.log()


class EDMPreconditioning(DenoiserPreconditioning):
    def __init__(self, sigma_data: float = 1.0):
        self.sigma_data = sigma_data

    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

    def get_c_in(self, sigma: Tensor) -> Tensor:
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return 0.25 * sigma.log()


class RectifiedFlowXLPreconditioning(DenoiserPreconditioning):
    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return torch.ones_like(sigma, device=sigma.device)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return -sigma

    def get_c_in(self, sigma: Tensor) -> Tensor:
        s_t = 1.0 / (1.0 + sigma)
        noise_std = ((1.0 / (sigma + 1.0)) ** 2.0 + (sigma / (sigma + 1.0)) ** 2.0) ** 0.5
        return s_t / noise_std

    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return 1000.0 * (sigma / (1 + sigma))


class RectifiedFlowComfyPreconditioning(DenoiserPreconditioning):
    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return torch.ones_like(sigma, device=sigma.device)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return -sigma

    def get_c_in(self, sigma: Tensor) -> Tensor:
        noise_std_inv = (sigma**2.0 + (1.0 - sigma) ** 2.0) ** -0.5
        return noise_std_inv

    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return 1000.0 * sigma
