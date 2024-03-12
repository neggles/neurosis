from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.modules.diffusion.model import Encoder
from neurosis.modules.diffusion.util import extract_into_tensor, make_beta_schedule
from neurosis.modules.regularizers import DiagonalGaussianDistribution


class LowScaleEncoder(nn.Module):
    # type annotations for the buffers
    betas: Tensor
    alphas_cumprod: Tensor
    alphas_cumprod_prev: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    log_one_minus_alphas_cumprod: Tensor
    sqrt_recip_alphas_cumprod: Tensor
    sqrt_recipm1_alphas_cumprod: Tensor

    def __init__(
        self,
        model: Encoder,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        timesteps: int = 1000,
        max_noise_level: int = 250,
        output_size: int = 64,
        scale_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_noise_level = max_noise_level
        self.model = model
        self.augmentation_schedule = self.register_schedule(
            timesteps=timesteps, linear_start=linear_start, linear_end=linear_end
        )
        self.out_size = output_size
        self.scale_factor = scale_factor

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps: int = 1000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-2,
        cosine_s: float = 8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

    def q_sample(self, x_start, t, noise: Optional[Tensor] = None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, x: Tensor):
        z = self.model.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        z = z * self.scale_factor
        noise_level = torch.randint(0, self.max_noise_level, (x.shape[0],), device=x.device).long()
        z = self.q_sample(z, noise_level)
        if self.out_size is not None:
            z = F.interpolate(z, size=self.out_size, mode="nearest")
        # z = z.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return z, noise_level

    def decode(self, z: Tensor):
        z = z / self.scale_factor
        return self.model.decode(z)
