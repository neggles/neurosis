import math
from typing import Callable, Optional

import torch
from scipy import integrate
from torch import Tensor

from neurosis.utils import append_dims, append_zero


def default_noise_sampler(x: Tensor) -> Callable[[Tensor, Tensor], Tensor]:
    def random_noise(sigma, sigma_next):
        return torch.randn_like(x)

    return random_noise


def linear_multistep_coeff(order: int, t: list[int], i: int, j: int, epsrel: float = 1e-4) -> float:
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def integrate_fn(tau) -> float:
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(integrate_fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(
    sigma_from: Tensor, sigma_to: Tensor, eta: Optional[float | Tensor] = 1.0
) -> tuple[Tensor, Tensor]:
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.min(sigma_to, eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma: Tensor) -> Tensor:
    return sigma.log().neg()


def to_sigma(neg_log_sigma: Tensor) -> Tensor:
    return neg_log_sigma.neg().exp()


def get_sigmas_vp(
    n: int,
    beta_d: float = 19.9,
    beta_min: float = 0.1,
    eps_s: float = 1e-3,
    device: torch.device | str = "cpu",
):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def get_sigmas_karras(
    n: int, sigma_min: float, sigma_max: float, rho: float = 7.0, device: torch.device | str = "cpu"
):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n: int, sigma_min: float, sigma_max: float, device: torch.device | str = "cpu"):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(
    n: int, sigma_min: float, sigma_max: float, rho: float = 1.0, device: torch.device | str = "cpu"
):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)
