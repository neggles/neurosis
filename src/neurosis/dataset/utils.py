from os import PathLike
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn

from neurosis.dataset.aspect.bucket import AspectBucket
from neurosis.utils.image import pil_ensure_rgb


class VAENormalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (x * 2.0) - 1.0


def clear_fsspec():
    import fsspec

    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None
    fsspec.asyn.reset_lock()


def set_s3fs_opts():
    import s3fs

    s3fs.S3FileSystem.retries = 10
    s3fs.S3FileSystem.connect_timeout = 30


def pil_crop_square(
    image: Image.Image,
    size: int | tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, tuple[int, int]]:
    if isinstance(size, int):
        size = (size, size)

    # crop short edge to match size
    image = ImageOps.cover(image, size, method=resampling)

    # crop long edge to match bucket long edge
    min_edge = min(image.size)

    delta_w, delta_h = image.size[0] - min_edge, image.size[1] - min_edge
    if all((delta_w, delta_h)):
        raise ValueError(f"Failed to crop short edge to match {size}!")

    top = np.random.randint(delta_h + 1)
    left = np.random.randint(delta_w + 1)
    image = image.crop((left, top, left + size[0], top + size[1]))

    return (image, (top, left))


def pil_crop_random(
    image: Image.Image,
    size: int | tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, tuple[int, int]]:
    if isinstance(size, int):
        size = (size, size)

    if image.size == size:
        return (image, (0, 0))

    # if too small upscale i guess
    if image.size[0] < size[0] or image.size[1] < size[1]:
        image = ImageOps.cover(image, size, method=Image.Resampling.LANCZOS)

    # downscale short edge to 2x the target size
    if image.size[0] > (size[0] * 2) and image.size[1] > (size[1] * 2):
        image = ImageOps.cover(image, (size[0] * 2, size[1] * 2), method=resampling)

    # now randomly crop the image to the desired size
    delta_w, delta_h = image.size[0] - size[0], image.size[1] - size[1]

    top, left = np.random.randint(delta_h + 1), np.random.randint(delta_w + 1)
    image = image.crop((left, top, left + size[0], top + size[1]))

    return (image, (top, left))


def load_crop_image_file(
    path: PathLike | bytes,
    resolution: int | tuple[int, int],
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, tuple[int, int]]:
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    path = Path(path).resolve()
    # load image
    image = Image.open(path)
    image = pil_ensure_rgb(image)
    return pil_crop_square(image, resolution, resampling)


def pil_crop_bucket(
    image: Image.Image,
    bucket: AspectBucket,
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, tuple[int, int]]:
    # resize short edge to match bucket short edge
    image = ImageOps.cover(image, bucket.size, method=resampling)
    width, height = image.size

    delta_w = width - bucket.width
    delta_h = height - bucket.height

    # this should never happen
    if all((delta_w != 0, delta_h != 0)):
        raise ValueError(f"Failed to crop short edge to match {bucket}!")

    # easy case, no cropping needed
    if delta_w == 0 and delta_h == 0:
        return (image, (0, 0))

    top, left = np.random.randint(delta_h + 1), np.random.randint(delta_w + 1)
    image = image.crop((left, top, left + bucket.width, top + bucket.height))
    return (image, (top, left))


def load_bucket_image_file(
    path: PathLike | bytes,
    bucket: AspectBucket,
    resampling: Image.Resampling = Image.Resampling.BICUBIC,
) -> tuple[Image.Image, tuple[int, int]]:
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    path = Path(path).resolve()
    # load image
    image = Image.open(path)
    image = pil_ensure_rgb(image)
    return pil_crop_bucket(image, bucket, resampling)


def clean_word(word_sep: str, word: str | bytes) -> str:
    if isinstance(word, (bytes, np.bytes_)):
        word = word.decode("utf-8")
    word = word.replace("_", word_sep)
    word = word.replace(" ", word_sep)
    return word.strip()


def collate_dict_lists(batch: list[dict]) -> dict[str, Tensor | np.ndarray]:
    # input will be a list of dicts, output should be a dict of lists
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [item[key] for item in batch]

    # now we do a torch.stack on any lists of tensors we have
    for k in batch_dict:
        if isinstance(batch_dict[k][0], Tensor):
            batch_dict[k] = torch.stack(batch_dict[k], dim=0)
        elif isinstance(batch_dict[k][0], np.ndarray):
            batch_dict[key] = np.stack(batch_dict[k], axis=0)

    return batch_dict


def collate_dict_stack(batch: dict[str, list]) -> dict[str, Tensor | np.ndarray | list]:
    """Collate function that takes a dict of lists and stacks any that are lists of tensors."""
    collated = {}
    for key, val in batch.items():
        if isinstance(val[0], Tensor):
            if val[0].ndim == 4 and val[0].shape[0] == 1:
                # images with batch dim, cat
                collated[key] = torch.cat(val, dim=0)
            elif val[0].ndim == 0:
                # scalars, stack
                collated[key] = torch.stack(val, dim=0)
            else:
                if val[0].shape[0] == 1:
                    # remove batch dim before stacking
                    collated[key] = torch.stack([x.squeeze(0) for x in val], dim=0)
                else:
                    # normal stack
                    collated[key] = torch.stack(val, dim=0)
        elif isinstance(val[0], (str, np.bytes_)):
            # this needs to be turned into a list of numpy arrays
            collated[key] = [np.array(x, dtype=np.bytes_) for x in val]
        else:
            # tuples or other, just pass through
            collated[key] = val

    return collated
