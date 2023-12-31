import numpy as np
import torch
from PIL import Image
from torch import Tensor


def numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    """
    Convert a numpy image or a batch of images to a list of PIL images.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def pil_to_numpy(images: list[Image.Image] | Image.Image) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.
    """
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images


def numpy_to_pt(images: np.ndarray) -> Tensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images


def pt_to_numpy(images: Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.
    """
    if isinstance(images, list):
        images = torch.stack(images, dim=0)

    if images.ndim == 4:
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # b, h, w, c
    else:
        images = images.cpu().permute(1, 2, 0).float().numpy()  # h, w, c
    return images


def pt_to_pil(images: Tensor) -> list[Image.Image]:
    """
    Convert a PyTorch tensor to a PIL image or list of PIL images.
    """

    images = pt_to_numpy(images)
    images = numpy_to_pil(images)
    return images


def pil_to_pt(images: list[Image.Image] | Image.Image) -> Tensor:
    """
    Convert a PIL image or a list of PIL images to a PyTorch tensor.
    """
    if not isinstance(images, list):
        images = [images]

    images = pil_to_numpy(images)
    images = numpy_to_pt(images)
    return images
