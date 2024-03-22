from abc import ABC

import torch

from neurosis.modules.diffusion.discretization import Discretization


class SigmaSampler(ABC):
    pass


class EDMSampling(SigmaSampler):
    def __init__(self, p_mean=-1.0, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        rand = rand if rand is not None else torch.randn((n_samples,))
        log_sigma = self.p_mean + self.p_std * (rand)
        return log_sigma.exp()


class DiscreteSampling(SigmaSampler):
    def __init__(
        self,
        discretization: Discretization,
        num_idx: int,
        do_append_zero: bool = True,
        flip: bool = True,
    ):
        self.num_idx = num_idx
        self.sigmas = discretization(num_idx, do_append_zero=do_append_zero, flip=flip)

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = rand or torch.randint(0, self.num_idx, (n_samples,))
        return self.idx_to_sigma(idx)
