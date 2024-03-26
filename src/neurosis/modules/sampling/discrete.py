import torch
from torch import Tensor

from neurosis.modules.diffusion.util import make_beta_schedule

from .common import DiffusionSampler2


class DiscreteSampler(DiffusionSampler2):
    def __init__(
        self,
        schedule: str = "linear",
        timesteps: int = 1000,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        cosine_s: float = 8e-3,
    ):
        super().__init__()

        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s

        betas = make_beta_schedule(
            schedule=schedule,
            n_timestep=timesteps,
            linear_start=self.linear_start,
            linear_end=self.linear_end,
            cosine_s=self.cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

        self.set_sigmas(sigmas, 1.0)

    def timestep(self, sigma: Tensor) -> Tensor:
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def sigma(self, timestep: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
        t = timestep.to(self.log_sigmas.device, torch.float32).clamp(0, len(self.sigmas) - 1)

        w = t.frac()
        sigma_low = (1 - w) * self.log_sigmas[t.floor().to(torch.int64)]
        sigma_high = w * self.log_sigmas[t.ceil().to(torch.int64)]

        sigma = torch.exp(sigma_low + sigma_high)
        return sigma.to(timestep.device, dtype=dtype)
