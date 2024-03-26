from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from neurosis.modules.diffusion.discretization import Discretization


class SigmaSampler(ABC):
    @abstractmethod
    def __call__(self, n_samples: int, rand: Optional[Tensor] = None):
        raise NotImplementedError("Abstract base class was called ;_;")


class EDMSampling(SigmaSampler):
    def __init__(
        self,
        p_mean: float = -1.0,
        p_std: float = 1.2,
    ):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples: int, rand: Optional[Tensor] = None):
        if rand is not None:
            rand = rand.to(torch.float32)
        else:
            rand = torch.randn((n_samples,), dtype=torch.float32)

        log_sigma = self.p_mean + self.p_std * rand
        return log_sigma.exp()


class DiscreteSampling(SigmaSampler):
    def __init__(
        self,
        discretization: Discretization,
        num_idx: int = 1000,
        do_append_zero: bool = True,
        flip: bool = True,
    ):
        self.num_idx = num_idx
        self.sigmas = discretization(num_idx, do_append_zero=do_append_zero, flip=flip)

    def idx_to_sigma(self, idx: int) -> Tensor:
        return self.sigmas[idx]

    def __call__(self, n_samples: int, rand: Optional[Tensor] = None):
        if rand is not None:
            idx = torch.clamp(rand.long(), 0, self.num_idx - 1)
        else:
            idx = torch.randint(0, self.num_idx, (n_samples,))
        return self.idx_to_sigma(idx)


class CosineSchedule(SigmaSampler):
    def __init__(
        self,
        s: float = 0.008,
        sigma_data: float = 1.0,
    ):
        self.s = torch.tensor([s])
        self.sigma_data = sigma_data
        self.min_var = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def __call__(
        self,
        n_samples: int,
        t: Optional[torch.Tensor] = None,
        shift: int = 1,
        return_logSNR: bool = False,
    ):
        if t is None:
            t = (1 - torch.rand(n_samples)).add(0.001).clamp(0.001, 1.0)
        s, min_var = self.s.to(t.device), self.min_var.to(t.device)
        var = torch.cos((s + t) / (1 + s) * torch.pi * 0.5).clamp(0, 1) ** 2 / min_var
        var = 0.0001 + var * 0.9999  # Modified variance calculation
        logSNR = (var / (1 - var)).log()

        if shift != 1:
            logSNR += 2 * np.log(1 / shift)

        if return_logSNR:
            return logSNR

        return torch.exp(-logSNR / 2) * self.sigma_data
