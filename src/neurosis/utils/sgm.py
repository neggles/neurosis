import logging
from textwrap import wrap as text_wrap
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.nn import Module

from neurosis.data import get_image_font

# generic wrapper typevar
T = TypeVar("T")

# used for overriding nn.Module methods while retaining type information
M = TypeVar("M", bound=Module)

logger = logging.getLogger(__name__)


def disabled_train(self: M, mode: bool = True) -> M:
    """Used to disable training on a module by monkeypatching the .train() method"""
    return self


def is_power_of_two(n) -> bool:
    return False if n <= 0 else bool((n & (n - 1)) == 0)


def autocast(f: Callable, enabled=True):
    def wrapper(*args, **kwargs):
        # Placeholder code
        pass

        # Call the original function
        return f(*args, **kwargs)

    return wrapper
    # def do_autocast(*args, **kwargs):
    #     with torch.cuda.amp.autocast(
    #         enabled=enabled,
    #         dtype=torch.get_autocast_gpu_dtype(),
    #         cache_enabled=torch.is_autocast_cache_enabled(),
    #     ):
    #         return f(*args, **kwargs)

    # return do_autocast


def log_txt_as_img(wh: tuple[int, int], xc: list[str], size: int = 10) -> Tensor:
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    width, height = wh

    # load font and get width
    font: ImageFont.FreeTypeFont = get_image_font(size=size)
    font_w = font.getlength(" ")
    nc = width // font_w  # number of characters per line

    for bi in range(b):
        # get text and wrap it at nc
        text_seq = xc[bi][0] if isinstance(xc[bi], list) else xc[bi]
        lines = "\n".join(text_wrap(text_seq, nc, tabsize=4))

        _, _, _, min_h = font.getbbox(lines)  # left, top, right, bottom
        if min_h > height:
            logger.warning(
                "Text too large to fit in image! Use a smaller font?"
                + f"Image size: {width}x{height}, Text size: {width}x{min_h}"
            )

        # make a new canvas and set up for drawing
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)

        try:
            # actually draw the text
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            logger.exception("Cant encode string for logging, skipping...")
            continue

        # convert to numpy and normalize
        txt = np.array(txt).transpose(2, 0, 1)  # H, W, C -> C, H, W
        txt = txt / 127.5 - 1.0  # 0-255 -> -1.0-1.0
        txts.append(txt)

    txts = torch.tensor(np.stack(txts))
    return txts


def ismap(x: Any) -> bool:
    if not isinstance(x, Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x: Any) -> bool:
    if not isinstance(x, Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def isheatmap(x: Any) -> bool:
    if not isinstance(x, Tensor):
        return False

    return x.ndim == 2


def isneighbors(x: Any) -> bool:
    if not isinstance(x, Tensor):
        return False
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


def expand_dims_like(x: Tensor, y: Tensor) -> Tensor:
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def mean_flat(tensor: Tensor) -> Tensor:
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model: Module, verbose: bool = False) -> int:
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        logger.info(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def append_zero(x: Tensor) -> Tensor:
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x: Tensor, ndim: int) -> Tensor:
    """Appends dimensions to the end of a tensor until it has ndim dimensions."""
    add_dims = ndim - x.ndim
    if add_dims < 0:
        raise ValueError(f"can't extend tensor from {x.ndim} to {ndim} dimensions!")
    return x[(...,) + (None,) * add_dims]


def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    """
    Will return the result of a recursive get attribute call.
    E.g.:
        a.b.c
        = getattr(getattr(a, "b"), "c")
        = get_nested_attribute(a, "b.c")
    If any part of the attribute call is an integer x with current obj a, will
    try to call a[x] instead of a.x first.
    """
    attributes = attribute_path.split(".")
    if depth is not None and depth > 0:
        attributes = attributes[:depth]
    assert len(attributes) > 0, "At least one attribute should be selected"
    current_attribute = obj
    current_key = None
    for level, attribute in enumerate(attributes):
        current_key = ".".join(attributes[: level + 1])
        try:
            id_ = int(attribute)
            current_attribute = current_attribute[id_]
        except ValueError:
            current_attribute = getattr(current_attribute, attribute)

    return (current_attribute, current_key) if return_key else current_attribute
