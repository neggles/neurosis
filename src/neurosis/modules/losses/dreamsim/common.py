from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def ensure_tuple(val: int | tuple[int, ...], n: int = 2) -> tuple[int, ...]:
    if isinstance(val, int):
        return (val,) * n
    elif len(val) != n:
        raise ValueError(f"Expected a tuple of {n} values, but got {len(val)}: {val}")
    return val


def use_fused_attn():
    if hasattr(F, "scaled_dot_product_attention"):
        return True
    return False


class QuickGELU(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


def get_act_layer(name: str) -> Callable[[], nn.Module]:
    match name:
        case "gelu":
            return nn.GELU
        case "quick_gelu":
            return QuickGELU
        case _:
            raise ValueError(f"Activation layer {name} not supported.")
