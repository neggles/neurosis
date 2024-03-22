from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class DenoiserScaling(ABC):
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


class EpsScaling(DenoiserScaling):
    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return torch.ones_like(sigma, device=sigma.device)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return (-sigma).clone()

    def get_c_in(self, sigma: Tensor) -> Tensor:
        return 1 / (sigma**2.0 + 1.0) ** 0.5

    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return sigma.clone()


class VScaling(EpsScaling):
    def __init__(self, sigma_data: float = 1.0):
        self.sigma_data = sigma_data

    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return self.sigma_data**2.0 / (sigma**2.0 + self.sigma_data)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return -sigma / ((sigma**2.0 + 1.0) ** 0.5)


class EDMScaling(VScaling):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def get_c_skip(self, sigma: Tensor) -> Tensor:
        return self.sigma_data**2.0 / (sigma**2.0 + self.sigma_data**2.0)

    def get_c_out(self, sigma: Tensor) -> Tensor:
        return sigma * (self.sigma_data / torch.sqrt(sigma**2.0 + self.sigma_data**2.0))

    def get_c_in(self, sigma: Tensor) -> Tensor:
        return 1 / (sigma**2.0 + self.sigma_data**2.0) ** 0.5

    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return 0.25 * sigma.log()


class VScalingWithEDMcNoise(VScaling):
    def get_c_noise(self, sigma: Tensor) -> Tensor:
        return 0.25 * sigma.log()
