import math

import torch
from torch import Tensor

from .common import DiffusionSampler2


class ContinuousEDMSampler(DiffusionSampler2):
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1000.0,
        sigma_data: float = 1.0,
    ):
        super().__init__()
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), 1000).exp()
        self.set_sigmas(sigmas, sigma_data)

    def timestep(self, sigma: Tensor) -> Tensor:
        return 0.25 * sigma.log()

    def sigma(self, timestep: int | float | Tensor) -> Tensor:
        return (timestep / 0.25).exp()

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 999999999.9
        if percent >= 1.0:
            return 0.0
        percent = 1.0 - percent

        log_sigma_min = math.log(self.sigma_min)
        return math.exp((math.log(self.sigma_max) - log_sigma_min) * percent + log_sigma_min)


class TanEDMSampler(ContinuousEDMSampler):
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1000.0,
        sigma_data: float = 1.0,
        eps: float = 5e-3,
    ):
        super().__init__()
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._eps = eps

        half_pi_t = torch.acos(torch.zeros(1, dtype=torch.float64)).squeeze(0)
        sigmas = torch.tan(torch.linspace(0.0, half_pi_t - eps, 1000, dtype=torch.float64)).to(torch.float32)
        self.set_sigmas(sigmas, sigma_data)
