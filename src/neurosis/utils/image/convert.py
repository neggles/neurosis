import numpy as np
import torch
from PIL import Image
from torch import Tensor


def numpy_to_pil(images: np.ndarray | list[np.ndarray], aslist: bool = False) -> list[Image.Image]:
    """
    Convert a numpy image or a batch of images to a list of PIL images.

    Inputs should be float32 in the range [0, 1] and N, H, W, C.
    If list of images, each should be H, W, C (no batch dimension)

    If a single image, returns a single PIL image.
    If a list of images, returns a list of PIL images.
    """
    if isinstance(images, list):
        # make func idempotent
        if isinstance(images[0], Image.Image):
            return images
        # we say you should pass in a list of HWC arrays, but we'll be nice
        # and allow a list of NHWC arrays too (as long as N=1)
        images = np.stack([x.squeeze(0) for x in images], axis=0)

    if np.isnan(images).any():
        raise ValueError("Images must not contain NaNs")
    if np.isinf(images).any():
        raise ValueError("Images must be finite")

    if images.dtype != np.float32:
        images = images.astype(np.float32)

    images = (images * 255.0).clip(0, 255).astype(np.uint8)

    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image, mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    if aslist is False:
        pil_images = pil_images[0] if len(pil_images) == 1 else pil_images

    return pil_images


def pil_to_numpy(images: Image.Image | list[Image.Image]) -> np.ndarray:
    """
    Convert a PIL image or a list of PIL images to NumPy arrays.

    Outputs will be float32 in the range [0, 1] and N, H, W, C.
    Single image will be returned as a batch of size 1.
    """
    if not isinstance(images, list):
        images = [images]

    images = [np.array(x, dtype=np.uint8) for x in images]
    images = np.stack(images, axis=0).astype(np.float32) / 255.0

    return images.squeeze()


def numpy_to_pt(images: np.ndarray | list[np.ndarray]) -> Tensor:
    """
    Convert a NumPy image to a PyTorch tensor.

    Inputs should be in the range [-1, 1] or [0, 1] and N, H, W, C.
    If a list of images, should be H, W, C (no batch dimension)

    Outputs will be in the range [-1, 1] or [0, 1] and N, C, H, W.
    Single image will be returned as a batch of size 1.

    No data type conversion or rescaling is performed so realistically the range doesn't matter.
    """
    if isinstance(images, list):
        # we say you should pass in a list of HWC arrays, but we'll be nice
        # and allow a list of NHWC arrays too (as long as N=1)
        images = np.stack([x.squeeze(0) for x in images], axis=0)

    if np.isnan(images).any() or np.isinf(images).any():
        raise ValueError("Images must be finite and not contain NaNs")

    if images.dtype != np.float32:
        images = images.astype(np.float32)

    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)

    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images)
    return images


def pt_to_numpy(images: Tensor | list[Tensor]) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Inputs should be in the range [-1, 1] or [0, 1] and N, C, H, W.
    If a list of images, should be H, W, C (no batch dimension)

    Outputs will be in the range [-1, 1] or [0, 1] and N, H, W, C.
    Single image will be returned as a batch of size 1.

    No data type conversion or rescaling is performed so realistically the range doesn't matter.
    """
    if isinstance(images, list):
        # we say you should pass in a list of CHW tensors, but we'll be nice
        # and allow a list of NCHW tensors too (as long as N=1)
        images = torch.stack([x.squeeze(0) for x in images], dim=0)

    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    images = images.cpu().permute(0, 2, 3, 1).numpy()
    return images


def pt_to_pil(images: Tensor | list[Tensor], aslist: bool = False) -> list[Image.Image]:
    """
    Convert a PyTorch tensor to a PIL image or list of PIL images.

    Inputs should be in the range [0, 1] and N, C, H, W
    If a list of images, each should be H, W, C (no batch dimension)

    If a single image, returns a single PIL image.
    If a list or batch of images, returns a list of PIL images.
    set aslist=True to force a list even if only one image is passed in.
    """
    if isinstance(images, list) and isinstance(images[0], Image.Image):
        return images

    images = pt_to_numpy(images)
    images = numpy_to_pil(images, aslist=aslist)
    return images


def pil_to_pt(images: Image.Image | list[Image.Image]) -> Tensor:
    """
    Convert a PIL image or a list of PIL images to a PyTorch tensor.

    Outputs will be float32 in the range [0, 1] and N, C, H, W
    Single image will be returned as a batch of size 1.
    """

    images = pil_to_numpy(images)
    images = numpy_to_pt(images)
    return images
