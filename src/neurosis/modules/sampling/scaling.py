import torch
from torch import Tensor

from .common import NoiseScaling


class EpsilonScaling(NoiseScaling):
    def calculate_input(self, sigma: Tensor, noise: Tensor) -> Tensor:
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma: Tensor, model_output: Tensor, model_input: Tensor) -> Tensor:
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma: Tensor, noise: Tensor, latents: Tensor, max_denoise: bool = False):
        if max_denoise:
            noise = noise * torch.sqrt(1.0 + sigma**2.0)
        else:
            noise = noise * sigma

        noise += latents
        return noise


class VScaling(EpsilonScaling):
    def calculate_denoised(self, sigma: Tensor, model_output: Tensor, model_input: Tensor) -> Tensor:
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (model_output.ndim - 1))

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

        return model_input * c_skip - model_output * c_out


class EDMScaling(VScaling):
    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (model_output.ndim - 1))

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

        return model_input * c_skip + model_output * c_out
