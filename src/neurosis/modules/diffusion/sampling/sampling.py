"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

import logging
from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor
from tqdm import tqdm

from neurosis.modules.guidance import Guider, IdentityGuider
from neurosis.utils import append_dims

from ..denoiser import Denoiser
from ..discretization import Discretization, RectifiedFlowComfyDiscretization
from .utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)

logger = logging.getLogger(__name__)


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization: Discretization,
        guider: Optional[Guider] = None,
        num_steps: Optional[int] = None,
        verbose: bool = False,
        device: str | torch.device = "cuda",
        rf_safeguard: bool = False,
    ):
        self.discretization = discretization
        self.guider = guider if guider is not None else IdentityGuider()
        self.num_steps = num_steps
        self.verbose = verbose
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.rf_safeguard = rf_safeguard
        self._comfy_rf = isinstance(self.discretization, RectifiedFlowComfyDiscretization)
        if self.rf_safeguard and not self._comfy_rf:
            logger.warning("RF safeguard is only available for ComfyRF! Continuing without it.")

    def prepare_sampling_loop(
        self,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
    ):
        num_steps = num_steps if num_steps is not None else self.num_steps
        if num_steps is None:
            raise ValueError(f"Step count must be set at init or call time! {self.num_steps=}")
        sigmas = self.discretization(num_steps)
        uc = uc if uc is not None else cond

        if self._comfy_rf:
            x *= sigmas[0]
        else:
            x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)

        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x: Tensor, denoiser, sigma, cond, uc):
        inputs = self.guider.prepare_inputs(x, sigma, cond, uc)
        denoised = denoiser(*inputs)
        denoised = self.guider(denoised, sigma)

        if self._comfy_rf and self.rf_safeguard:
            # normalized output hack for the start of transitioning phase
            # !! only works for ComfyRF for now
            sigma = append_dims(sigma, x.ndim)
            alpha = 1.0 - sigma
            denoised_x0 = denoised / alpha
            std_values = denoised_x0.std(dim=tuple(range(1, denoised.dim())))
            mask_lower = std_values < 0.5
            mask_upper = std_values > 1.5
            mask = mask_lower | mask_upper
            denoised[mask] /= std_values[mask].view(-1, *[1] * (denoised.dim() - 1))

        return denoised

    def get_sigma_gen(self, num_sigmas: int):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            logger.info("#" * 30, " Sampling setting ", "#" * 30)
            logger.info(f"Sampler: {self.__class__.__name__}")
            logger.info(f"Discretization: {self.discretization.__class__.__name__}")
            logger.info(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator

    @abstractmethod
    def __call__(
        self,
        denoiser: Denoiser,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError("Abstract base class was called ;_;")


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    @abstractmethod
    def sampler_step(
        self,
        sigma: Tensor | float,
        next_sigma: Tensor | float,
        denoiser: Denoiser,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Abstract base class was called ;_;")

    def euler_step(self, x: Tensor, d: Tensor, dt: Tensor):
        """it's just an FMA, stop overthinking it"""
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(
        self,
        sigma: Tensor,
        next_sigma: Tensor,
        denoiser: Denoiser,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        gamma: float = 0.0,
    ):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def __call__(
        self,
        denoiser: Denoiser,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(
        self,
        eta: float = 1.0,
        s_noise: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(
        self,
        x: Tensor,
        denoised: Tensor,
        sigma: Tensor,
        sigma_down: Tensor,
    ):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(
        self,
        x: Tensor,
        sigma: Tensor | float,
        next_sigma: Tensor,
        sigma_up: Tensor,
    ):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(
        self,
        denoiser: Denoiser,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(
        self,
        denoiser: Denoiser,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), "D", **kwargs)
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step)
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        return x, denoised

    def __call__(
        self,
        denoiser: Denoiser,
        x: Tensor,
        cond: Tensor,
        uc: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x
