import torch
from torch import Tensor

from .common import DiffusionSampler2, SigmaScheduler


class SimpleScheduler(SigmaScheduler):
    def get_schedule(self, n_steps: int, dtype: torch.dtype = torch.float32) -> Tensor:
        stride = len(self.sampler.sigmas) / n_steps

        schedule = []
        for x in range(n_steps):
            sigma_offs = int(x * stride) + 1
            schedule.append(self.sampler.sigmas[-sigma_offs].item())

        schedule.append(0.0)
        return torch.tensor(schedule, dtype=dtype)


class DDIMScheduler(SigmaScheduler):
    def get_schedule(self, n_steps: int, dtype: torch.dtype = torch.float32) -> Tensor:
        schedule = []
        stride = max(len(self.sampler.sigmas) // n_steps, 1)
        x = 1
        while x < len(self.sampler.sigmas):
            schedule.append([float(self.sampler.sigmas[x])])
            x += stride
        schedule = schedule[::-1]

        schedule.append(0.0)
        return torch.tensor(schedule, dtype=dtype)


class UniformScheduler(SigmaScheduler):
    def get_schedule(self, n_steps: int, dtype: torch.dtype = torch.float32) -> Tensor:
        start = self.sampler.timestep(self.sampler.sigma_max)
        end = self.sampler.timestep(self.sampler.sigma_min)

        timen_steps = torch.linspace(start, end, n_steps)

        schedule = []
        for x in range(len(timen_steps)):
            ts = timen_steps[x]
            schedule.append(self.sampler.sigma(ts))

        schedule.append(0.0)
        return torch.tensor(schedule, dtype=dtype)


class SGMUniformScheduler(SigmaScheduler):
    def get_schedule(self, n_steps: int, dtype: torch.dtype = torch.float32) -> Tensor:
        start = self.sampler.timestep(self.sampler.sigma_max)
        end = self.sampler.timestep(self.sampler.sigma_min)

        timen_steps = torch.linspace(start, end, n_steps + 1)[:-1]

        schedule = []
        for x in range(len(timen_steps)):
            ts = timen_steps[x]
            schedule.append(self.sampler.sigma(ts))

        schedule.append(0.0)
        return torch.tensor(schedule, dtype=dtype)


def get_sigma_scheduler(name: str, sampler: DiffusionSampler2) -> SigmaScheduler:
    match name:
        case "simple":
            return SimpleScheduler(sampler)
        case "ddim":
            return DDIMScheduler(sampler)
        case "uniform":
            return UniformScheduler(sampler)
        case "sgm_uniform":
            return SGMUniformScheduler(sampler)
        case _:
            raise ValueError(f"Unknown scheduler {name}")
