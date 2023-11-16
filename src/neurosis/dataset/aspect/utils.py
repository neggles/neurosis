from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import v2 as T

from neurosis.dataset.aspect.bucket import AspectBucket


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


def pil_crop_bucket(
    image: Image.Image,
    bucket: AspectBucket,
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, tuple[int, int]]:
    # resize short edge to match bucket short edge
    image = ImageOps.cover(image, bucket.size, method=resampling)

    # crop long edge to match bucket long edge
    min_edge = min(image.size)

    delta_w = image.size[0] - min_edge
    delta_h = image.size[1] - min_edge
    if all([delta_w, delta_h]):
        raise ValueError(f"Failed to crop short edge to match {bucket}!")

    top = np.random.randint(delta_h + 1)
    left = np.random.randint(delta_w + 1)
    image = T.functional.crop(image, top, left, bucket.height, bucket.width)
    return image, (top, left)


def load_bucket_image_file(
    path: PathLike | bytes,
    bucket: AspectBucket,
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Tensor, tuple[int, int]]:
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    path = Path(path).resolve()
    # load image
    image = Image.open(path)
    image = pil_ensure_rgb(image)
    image, (top, left) = pil_crop_bucket(image, bucket, resampling)
    return image, (top, left)
