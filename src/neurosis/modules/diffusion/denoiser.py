from typing import Callable

from torch import Tensor, nn

from neurosis.modules.diffusion.discretizer import Discretization
from neurosis.utils import append_dims


class Denoiser(nn.Module):
    def __init__(self, weighting: Callable, scaling: Callable):
        super().__init__()

        self.weighting = weighting
        self.scaling = scaling

    def possibly_quantize_sigma(self, sigma):
        return sigma

    def possibly_quantize_c_noise(self, c_noise):
        return c_noise

    def w(self, sigma):
        return self.weighting(sigma)

    def __call__(self, network: nn.Module, input: Tensor, sigma: Tensor, cond: Tensor):
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        return network(input * c_in, c_noise, cond) * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    sigmas: Tensor

    def __init__(
        self,
        weighting: Callable,
        scaling: Callable,
        num_idx: int,
        discretization: Discretization,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(weighting, scaling)
        sigmas = discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas)
        self.quantize_c_noise = quantize_c_noise

    def sigma_to_idx(self, sigma: Tensor) -> Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx) -> Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: Tensor) -> Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: Tensor) -> Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
