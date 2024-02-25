from enum import Enum
from typing import TypeAlias

import numpy as np
import torch
from PIL import Image
from torch import Tensor

# type aliases
LogDictType: TypeAlias = dict[str, Tensor | np.ndarray | Image.Image | str | float | int]
BatchDictType: TypeAlias = dict[str, Tensor | np.ndarray | str]


# custom types
class StepType(str, Enum):
    global_step = "global_step"  # default
    batch_idx = "batch_idx"  # batch index instead of global step
    global_batch = "global_batch"  # global step * accumulate_grad_batches
    sample_idx = "sample_idx"  # global step * accumulate_grad_batches * batch_size


# compiled convenience functions
@torch.compile
def diff_images(
    inputs: Tensor,
    recons: Tensor,
    boost: float = 3.0,
) -> Tensor:
    diff = torch.clamp(recons, -1.0, 1.0).sub(inputs).abs().mul(0.5)

    boosted = diff.mul(boost).clamp(0.0, 1.0).mul(2.0).sub(1.0)
    diff = diff.mul(2.0).sub(1.0)

    return diff.contiguous(), boosted.contiguous()
