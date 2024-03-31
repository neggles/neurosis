from abc import ABC, abstractmethod
from math import log

import numpy as np
import torch
from torch import Tensor

from neurosis.utils import append_zero

from .util import make_beta_schedule


def generate_roughly_equally_spaced_steps(num_substeps: int, max_step: int) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class Discretization(ABC):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        n: int,
        do_append_zero: bool = False,
        device: str | torch.device = "cpu",
        flip: bool = False,
    ):
        sigmas: Tensor = self.get_sigmas(n, device=device)

        if do_append_zero:
            sigmas = append_zero(sigmas)
        if flip:
            sigmas = sigmas.flip((0,))

        return sigmas

    @abstractmethod
    def get_sigmas(self, n: int, device: str | torch.device) -> Tensor:
        raise NotImplementedError("Abstract base class was called ;_;")


class EDMcDiscretization(Discretization):
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1000.0,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        sigmas = torch.linspace(log(self.sigma_min), log(self.sigma_max), n, dtype=torch.float32).exp()

        # return flipped so largest sigma is first
        return sigmas.flip(0).to(device)


class EDMcSimpleDiscretization(Discretization):
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1000.0,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        sigmas = torch.linspace(log(self.sigma_min), log(self.sigma_max), 1000, dtype=torch.float32).exp()
        sigs = []
        ss = len(sigmas) / n
        for x in range(n):
            sigs += [float(sigmas[-(1 + int(x * ss))])]
        sigs += [0.0]
        sigs = torch.tensor(sigs)

        return sigs.to(device)


class TanZeroSNRDiscretization(Discretization):
    def __init__(self, sigma_max: float = 9e4):
        super().__init__()
        self.sigma_max = sigma_max

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        # these calcs need to be float64 or they'll overflow in intermediate steps
        half_pi_t = torch.acos(torch.zeros(1, dtype=torch.float64)).squeeze(0)
        sigmas = torch.tan(torch.linspace(0.0, half_pi_t - 0.001, 1000, dtype=torch.float64))
        # return flipped so largest sigma is first and cast to float32
        return sigmas.flip(0).to(device, dtype=torch.float32)


class EDMDiscretization(Discretization):
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> Tensor:
        ramp = torch.linspace(0, 1, n, device=device, dtype=torch.float32)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        # largest sigma is already first here
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
        self.alphas = 1.0 - make_beta_schedule("linear", num_timesteps, linear_start, linear_end)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0, dtype=torch.float32)

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
