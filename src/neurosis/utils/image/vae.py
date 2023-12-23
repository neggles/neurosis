import numpy as np
from torch import Tensor


def normalize(images: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """
    Normalize an image array to [-1,1].
    """
    return 2.0 * images - 1.0


def denormalize(images: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)
