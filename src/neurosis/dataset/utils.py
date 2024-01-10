from os import PathLike
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from torch import Tensor

from neurosis.dataset.aspect.bucket import AspectBucket
from neurosis.utils.image import pil_ensure_rgb


def clear_fsspec():
    import fsspec

    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None


def set_s3fs_opts():
    import s3fs

    s3fs.S3FileSystem.retries = 10
    s3fs.S3FileSystem.connect_timeout = 30


def pil_crop_square(
    image: Image.Image,
    size: int | tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
    return_dict: bool = False,
) -> Image.Image:
    if isinstance(size, int):
        size = (size, size)

    # crop short edge to match size
    image = ImageOps.cover(image, size, method=resampling)

    # crop long edge to match bucket long edge
    min_edge = min(image.size)

    delta_w = image.size[0] - min_edge
    delta_h = image.size[1] - min_edge
    if all([delta_w, delta_h]):
        raise ValueError(f"Failed to crop short edge to match {size}!")

    top = np.random.randint(delta_h + 1)
    left = np.random.randint(delta_w + 1)
    image = image.crop((left, top, left + size[0], top + size[1]))

    if return_dict:
        return {"image": image, "crop_coords_top_left": (top, left)}
    return (image, (top, left))


def pil_crop_bucket(
    image: Image.Image,
    bucket: AspectBucket,
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
    return_dict: bool = False,
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
    image = image.crop((left, top, left + bucket.width, top + bucket.height))

    if return_dict:
        return {"image": image, "crop_coords_top_left": (top, left)}
    return (image, (top, left))


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


def clean_word(word_sep: str, word: str | bytes) -> str:
    if isinstance(word, (bytes, np.bytes_)):
        word = word.decode("utf-8")
    word = word.replace("_", word_sep)
    word = word.replace(" ", word_sep)
    return word.strip()
