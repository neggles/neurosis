from abc import ABC

import torch
from torch import Tensor


class DenoiserWeighting(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, sigma: Tensor) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")


class UnitWeighting(DenoiserWeighting):
    def __call__(self, sigma: Tensor) -> Tensor:
        weights = torch.ones_like(sigma, device=sigma.device)
        return weights


class EpsWeighting(DenoiserWeighting):
    def __call__(self, sigma: Tensor) -> Tensor:
        weights = sigma**-2.0
        return weights


class EDMWeighting(DenoiserWeighting):
    def __init__(self, sigma_data: float = 1.0):
        super().__init__()
        self.sigma_data = sigma_data

    def __call__(self, sigma: Tensor) -> Tensor:
        weights = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return weights
