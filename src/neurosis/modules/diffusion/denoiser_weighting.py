from abc import ABC

import torch
from torch import Tensor


class DenoiserWeighting(ABC):
    def __call__(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")


class UnitWeighting(DenoiserWeighting):
    def __call__(self, sigma: Tensor) -> Tensor:
        return torch.ones_like(sigma, device=sigma.device)


class EDMWeighting(DenoiserWeighting):
    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma: Tensor) -> Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting(DenoiserWeighting):
    def __call__(self, sigma):
        return sigma**-2.0
