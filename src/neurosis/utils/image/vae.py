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
    return images + 1.0 / 2.0


def is_image_tensor(x: np.ndarray | Tensor) -> bool:
    if x.ndim == 3 and x.shape[0] == 3:
        return True
    if x.ndim == 4 and x.shape[1] == 3:
        return True
    return False
