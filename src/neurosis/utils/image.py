import numpy as np
import torch
from PIL import Image
from torch import Tensor


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(
    image: Image.Image,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), fill)
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def numpy_to_pil(images: np.ndarray) -> list[Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image.
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
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def pt_to_pil(images: Tensor) -> list[Image.Image]:
    """
    Convert a PyTorch tensor to a PIL image.
    """
    images = pt_to_numpy(images)
    images = numpy_to_pil(images)
    return images


def pil_to_pt(images: list[Image.Image] | Image.Image) -> Tensor:
    """
    Convert a PIL image to a PyTorch tensor.
    """
    if not isinstance(images, list):
        images = [images]

    images = pil_to_numpy(images)
    images = numpy_to_pt(images)
    return images


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
