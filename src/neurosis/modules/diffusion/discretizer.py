import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from neurosis.modules.diffusion.util import make_beta_schedule
from neurosis.utils import append_zero


def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class Discretization(ABC):
    def __call__(
        self,
        n: int,
        do_append_zero: bool = True,
        device: str | torch.device = "cpu",
        flip: bool = False,
    ):
        sigmas = self.get_sigmas(n, device=device)

        if do_append_zero:
            sigmas = append_zero(sigmas)
        if flip:
            sigmas = torch.flip(sigmas, (0,))

        return sigmas

    @abstractmethod
    def get_sigmas(self, n: int, device: str | torch.device) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")


class EDMcDiscretization(Discretization):
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 120,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        sigmas = torch.linspace(math.log(self.sigma_min), math.log(self.sigma_max), n).exp()
        return sigmas.flip(0).to(device, dtype=torch.float32)


class EDMDiscretization(Discretization):
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas


class LegacyDDPMDiscretization(Discretization):
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        alphas = 1.0 - make_beta_schedule("linear", num_timesteps, linear_start, linear_end)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps].clone().detach().requires_grad_(True)
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod.clone().detach().requires_grad_(True)
        else:
            raise ValueError(f"n ({n}) must be less than or equal to num_timesteps ({self.num_timesteps})")

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return sigmas.flip(0).to(device, dtype=torch.float32)
