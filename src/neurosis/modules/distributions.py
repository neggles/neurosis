from typing import Any, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError("Abstract base class was called ;_;")

    def mode(self):
        raise NotImplementedError("Abstract base class was called ;_;")


class DiracDistribution(AbstractDistribution):
    def __init__(self, value) -> None:
        self.value = value

    def sample(self) -> Any:
        return self.value

    def mode(self) -> Any:
        return self.value


class DiagonalGaussianDistribution(AbstractDistribution):
    def __init__(self, parameters: Union[Parameter, list[Parameter]], deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self) -> Tensor:
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other: Optional[Tensor] = None) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: Tensor, dims=[1, 2, 3]) -> Tensor:
        if self.deterministic:
            return torch.tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> Tensor:
        return self.mean
