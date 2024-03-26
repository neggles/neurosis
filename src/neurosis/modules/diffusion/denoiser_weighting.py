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


# class VWeighting(EDMWeighting):
#     def __init__(self):
#         super().__init__(sigma_data=1.0)


# class MinSNRGammaVWeighting(VWeighting):
#     def __init__(self, gamma: float = 5.0):
#         super().__init__()
#         self.gamma = gamma

#     def __call__(self, sigma: Tensor) -> Tensor:
#         # get regular v-pred weights
#         v_weights = super().__call__(sigma)
#         # calculate SNR, assuming unit variance
#         snr = 1 / sigma**2
#         # get min(snr, gamma) for each element
#         min_snr_gamma = torch.min(snr, torch.full_like(snr, self.gamma))
#         # prevent div/0 errors with v-prediction
#         snr_weight = min_snr_gamma.div(snr + 1)
#         # return the adjusted weights
#         return v_weights * snr_weight
