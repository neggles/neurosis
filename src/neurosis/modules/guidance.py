from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor


class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale):
        return uncond + scale * (cond - uncond)


class DiffusionGuider(ABC):
    @abstractmethod
    def __call__(self, x: Tensor, sigma) -> Tensor:
        raise NotImplementedError("Abstract base class was called")

    @abstractmethod
    def prepare_inputs(self, x: Tensor, s, c, uc):
        raise NotImplementedError("prepare_inputs() not implemented!")


class VanillaCFG(DiffusionGuider):
    """
    implements parallelized CFG
    """

    def __init__(
        self,
        scale: float = 1.0,
        dyn_thresh: Optional[Callable] = None,
    ):
        self.scale = scale
        self.dyn_thresh = dyn_thresh if dyn_thresh is not None else NoDynamicThresholding()

    def __call__(self, x: Tensor, sigma) -> Tensor:
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def scale_schedule(self, sigma) -> float:
        return self.scale

    def prepare_inputs(self, x: Tensor, s, c, uc):
        c_out = dict()

        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out


class IdentityGuider(DiffusionGuider):
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out
