from typing import Optional

import torch
from scipy import integrate
from torch import Tensor

from neurosis.utils import append_dims


def linear_multistep_coeff(order, t, i, j, epsrel: float = 1e-4) -> float:
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
    if not eta:
        return sigma_to, Tensor(0.0)
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma: Tensor) -> Tensor:
    return sigma.log().neg()


def to_sigma(neg_log_sigma: Tensor) -> Tensor:
    return neg_log_sigma.neg().exp()
