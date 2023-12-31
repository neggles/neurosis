import logging
from typing import Union

from torch import Tensor, nn

from neurosis.utils import append_dims

from .denoiser_scaling import DenoiserScaling
from .discretizer import Discretization

logger = logging.getLogger(__name__)


class Denoiser(nn.Module):
    def __init__(self, scaling: DenoiserScaling):
        super().__init__()

        self.scaling: DenoiserScaling = scaling

    def possibly_quantize_sigma(self, sigma: Tensor) -> Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: Tensor) -> Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        input: Tensor,
        sigma: Tensor,
        cond: dict,
        **additional_model_inputs,
    ) -> Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape
        sigma = append_dims(sigma, input.ndim)
        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))

        net_input = input * c_in
        net_output = network(net_input, c_noise, cond, **additional_model_inputs)

        return net_output * c_out + input * c_skip


class DiscreteDenoiser(Denoiser):
    sigmas: Tensor

    def __init__(
        self,
        scaling: DenoiserScaling,
        num_idx: int,
        discretization: Discretization,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = True,
    ):
        super().__init__(scaling)
        sigmas = discretization(num_idx, do_append_zero=do_append_zero, flip=flip)
        self.register_buffer("sigmas", sigmas, persistent=False)
        self.quantize_c_noise = quantize_c_noise
        self.num_idx = num_idx

    def sigma_to_idx(self, sigma: Tensor) -> Tensor:
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def idx_to_sigma(self, idx: Union[Tensor, int]) -> Tensor:
        return self.sigmas[idx]

    def possibly_quantize_sigma(self, sigma: Tensor) -> Tensor:
        return self.idx_to_sigma(self.sigma_to_idx(sigma))

    def possibly_quantize_c_noise(self, c_noise: Tensor) -> Tensor:
        if self.quantize_c_noise:
            return self.sigma_to_idx(c_noise)
        else:
            return c_noise
