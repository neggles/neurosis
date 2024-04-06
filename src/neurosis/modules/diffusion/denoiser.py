import logging
from typing import Union

from torch import Tensor, nn

from neurosis.utils import append_dims

from .denoiser_preconditioning import DenoiserPreconditioning
from .discretization import Discretization

logger = logging.getLogger(__name__)


class Denoiser(nn.Module):
    def __init__(
        self,
        preconditioning: DenoiserPreconditioning,
    ):
        super().__init__()
        self.preconditioning: DenoiserPreconditioning = preconditioning

    def possibly_quantize_sigma(self, sigma: Tensor) -> Tensor:
        return sigma

    def possibly_quantize_c_noise(self, c_noise: Tensor) -> Tensor:
        return c_noise

    def forward(
        self,
        network: nn.Module,
        inputs: Tensor,
        sigma: Tensor,
        cond: dict,
        output_mode="D",
        **additional_model_inputs,
    ) -> Tensor:
        sigma = self.possibly_quantize_sigma(sigma)
        sigma_shape = sigma.shape

        sigma = append_dims(sigma, inputs.ndim)
        c_skip, c_out, c_in, c_noise = self.preconditioning(sigma)

        c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
        c_in = c_in.to(inputs.dtype)
        c_out = c_out.to(inputs.dtype)
        c_skip = c_skip.to(inputs.dtype)
        c_out = c_out.to(inputs.dtype)

        net_inputs = inputs * c_in
        net_outputs = network(net_inputs, c_noise, cond, **additional_model_inputs)
        match output_mode:
            case "D":
                return net_outputs * c_out + inputs * c_skip
            case "F":
                return net_outputs
            case _:
                return net_outputs * c_out + inputs * c_skip


class DiscreteDenoiser(Denoiser):
    sigmas: Tensor
    log_sigmas: Tensor

    def __init__(
        self,
        preconditioning: DenoiserPreconditioning,
        num_idx: int,
        discretization: Discretization,
        do_append_zero: bool = False,
        quantize_c_noise: bool = True,
        flip: bool = False,
    ):
        super().__init__(preconditioning)
        self.num_idx = num_idx
        self.quantize_c_noise = quantize_c_noise
        self.do_append_zero = do_append_zero
        self.flip = flip

        sigmas = discretization(self.num_idx, do_append_zero=self.do_append_zero, flip=self.flip)
        self.register_buffer("sigmas", sigmas, persistent=False)
        self.register_buffer("log_sigmas", sigmas.log(), persistent=False)

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
