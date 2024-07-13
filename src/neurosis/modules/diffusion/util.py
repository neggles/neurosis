"""
partially adopted from
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
and
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
and
https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py

thanks!
"""

import math
from functools import wraps
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
    cosine_s: float = 8e-3,
) -> Tensor:
    match schedule:
        case "linear":
            betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
        case "cosine":
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = torch.clamp(betas, min=0, max=0.999)
        case "sqrt_linear":
            betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
        case "sqrt":
            betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
        case _:
            raise ValueError(f"unknown schedule: {schedule}")

    return betas


def extract_into_tensor(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def mixed_checkpoint(func, inputs: dict, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass. This differs from the original checkpoint function
    borrowed from https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py in that
    it also works with non-tensor inputs
    :param func: the function to evaluate.
    :param inputs: the argument dictionary to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        tensor_keys = [key for key in inputs if isinstance(inputs[key], Tensor)]
        tensor_inputs = [inputs[key] for key in inputs if isinstance(inputs[key], Tensor)]
        non_tensor_keys = [key for key in inputs if not isinstance(inputs[key], Tensor)]
        non_tensor_inputs = [inputs[key] for key in inputs if not isinstance(inputs[key], Tensor)]
        args = tuple(tensor_inputs) + tuple(non_tensor_inputs) + tuple(params)
        return MixedCheckpointFunction.apply(
            func,
            len(tensor_inputs),
            len(non_tensor_inputs),
            tensor_keys,
            non_tensor_keys,
            *args,
        )
    else:
        return func(**inputs)


class MixedCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable,
        length_tensors: int,
        length_non_tensors: int,
        tensor_keys: Sequence[str],
        non_tensor_keys: Sequence[str],
        *args,
    ):
        ctx.end_tensors = length_tensors
        ctx.end_non_tensors = length_tensors + length_non_tensors
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        if len(tensor_keys) != length_tensors:
            raise ValueError("MixedCheckpointFunction: incorrect number of tensor keys")
        if len(non_tensor_keys) != length_non_tensors:
            raise ValueError("MixedCheckpointFunction: incorrect number of non-tensor keys")

        ctx.input_tensors = {key: val for (key, val) in zip(tensor_keys, list(args[: ctx.end_tensors]))}
        ctx.input_non_tensors = {
            key: val for (key, val) in zip(non_tensor_keys, list(args[ctx.end_tensors : ctx.end_non_tensors]))
        }
        ctx.run_function = run_function
        ctx.input_params = list(args[ctx.end_non_tensors :])

        with torch.no_grad():
            output_tensors = ctx.run_function(**ctx.input_tensors, **ctx.input_non_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        # additional_args = {key: ctx.input_tensors[key] for key in ctx.input_tensors if not isinstance(ctx.input_tensors[key],Tensor)}
        ctx.input_tensors = {
            key: ctx.input_tensors[key].detach().requires_grad_(True) for key in ctx.input_tensors
        }

        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = {
                key: ctx.input_tensors[key].view_as(ctx.input_tensors[key]) for key in ctx.input_tensors
            }
            # shallow_copies.update(additional_args)
            output_tensors = ctx.run_function(**shallow_copies, **ctx.input_non_tensors)
        input_grads = torch.autograd.grad(
            output_tensors,
            list(ctx.input_tensors.values()) + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (
            (None, None, None, None, None)
            + input_grads[: ctx.end_tensors]
            + (None,) * (ctx.end_non_tensors - ctx.end_tensors)
            + input_grads[ctx.end_tensors :]
        )


def timestep_embedding(
    timesteps: Tensor,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> Tensor:
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def scale_module(module: nn.Module, scale) -> nn.Module:
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor: Tensor) -> Tensor:
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


@wraps(nn.modules.conv._ConvNd)
def conv_nd(dims, *args, **kwargs) -> nn.Conv1d | nn.Conv2d | nn.Conv3d:
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    match dims:
        case 1:
            return nn.Conv1d(*args, **kwargs)
        case 2:
            return nn.Conv2d(*args, **kwargs)
        case 3:
            return nn.Conv3d(*args, **kwargs)
        case _:
            raise ValueError(f"unsupported dimensions: {dims}")


@wraps(nn.modules.pooling._AvgPoolNd)
def avg_pool_nd(dims: int, *args, **kwargs) -> nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d:
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    match dims:
        case 1:
            return nn.AvgPool1d(*args, **kwargs)
        case 2:
            return nn.AvgPool2d(*args, **kwargs)
        case 3:
            return nn.AvgPool3d(*args, **kwargs)
        case _:
            raise ValueError(f"unsupported dimensions: {dims}")


class AlphaBlender(nn.Module):
    strategies = ["learned", "fixed", "learned_with_images"]
    mix_factor: Tensor

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        rearrange_pattern: str = "b t -> (b t) 1 1",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.rearrange_pattern = rearrange_pattern

        assert merge_strategy in self.strategies, f"merge_strategy needs to be in {self.strategies}"

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", Tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", nn.Parameter(Tensor([alpha])))
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: Tensor) -> Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            assert image_only_indicator is not None, "need image_only_indicator ..."
            alpha = torch.where(
                image_only_indicator.bool(),
                torch.ones(1, 1, device=image_only_indicator.device),
                rearrange(torch.sigmoid(self.mix_factor), "... -> ... 1"),
            )
            alpha = rearrange(alpha, self.rearrange_pattern)
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")
        return alpha

    def forward(
        self,
        x_spatial: Tensor,
        x_temporal: Tensor,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x
