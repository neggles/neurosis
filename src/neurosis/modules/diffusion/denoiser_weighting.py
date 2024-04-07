from abc import ABC, abstractmethod

import torch
from torch import Tensor


class DenoiserWeighting(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
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


class RectifiedFlowWeighting(DenoiserWeighting):
    def __init__(self, m: float = 0.0, s: float = 1.0):
        super().__init__()
        self.m = m
        self.s = s

    def __call__(self, sigma: Tensor) -> Tensor:
        sigma = sigma.to(torch.float64)
        t = sigma / (1.0 + sigma)
        cfm_weights = 1 / (1 - t) ** 2
        # logit-normal sampling
        half_pi = torch.acos(torch.zeros(1, dtype=torch.float64))[0]
        pi_weights = (
            (1 / (self.s * (4.0 * half_pi) ** 0.5))
            * (1 / (t * (1.0 - t)))
            * torch.exp(-0.5 * (torch.log(sigma) - self.m) ** 2 / self.s**2)
        )
        return cfm_weights * pi_weights


class RectifiedFlowComfyWeighting(DenoiserWeighting):
    def __init__(self, m: float = 0.0, s: float = 1.0):
        super().__init__()
        self.m = m
        self.s = s

    def __call__(self, sigma: Tensor) -> Tensor:
        t = sigma.to(torch.float64)

        cfm_weights = 1 / (1 - t) ** 2
        # logit-normal sampling
        half_pi = torch.acos(torch.zeros(1, dtype=torch.float64))[0]
        pi_weights = (
            (1 / (self.s * (4.0 * half_pi) ** 0.5))
            * (1 / (t * (1.0 - t)))
            * torch.exp(-0.5 * (torch.log(t / (1 - t)) - self.m) ** 2 / self.s**2)
        )
        return cfm_weights * pi_weights


class MinSNRGammaModifier(DenoiserWeighting):
    def __init__(
        self,
        weighting: DenoiserWeighting,
        gamma: float = 5,
        v_pred: bool = False,
    ):
        super().__init__()
        self.weighting = weighting
        self.gamma = gamma
        self.v_pred = v_pred

    def __call__(self, sigma: Tensor) -> Tensor:
        weights = self.weighting(sigma)
        snr = 1.0 / sigma**2

        # get min(snr, gamma) for each element
        snr_weight = torch.min(snr, torch.full_like(snr, self.gamma))
        if self.v_pred:
            snr_weight = snr_weight.div(snr + 1.0)
        else:
            snr_weight = snr_weight.div(snr)

        return weights * snr_weight
