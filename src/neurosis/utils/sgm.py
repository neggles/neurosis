import importlib
import logging
from functools import partial
from os import PathLike
from pathlib import Path
from textwrap import wrap as text_wrap
from typing import Any, Callable, TypeVar

import fsspec
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file as load_safetensors
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


def get_string_from_tuple(s: str):
    try:
        # Check if the string starts and ends with parentheses
        if s[0] == "(" and s[-1] == ")":
            # Convert the string to a tuple
            t = eval(s)
            # Check if the type of t is tuple
            if type(t) == tuple:
                return t[0]
            else:
                pass
    except Exception:
        pass
    return s


def is_power_of_two(n) -> bool:
    return False if n <= 0 else bool((n & (n - 1)) == 0)


def autocast(f: Callable, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,
            dtype=torch.get_autocast_gpu_dtype(),
            cache_enabled=torch.is_autocast_cache_enabled(),
        ):
            return f(*args, **kwargs)

    return do_autocast


def load_partial_from_config(config) -> partial[Any]:
    if "class_path" in config:
        # PyTorch Lightning syntax
        target = config["class_path"]
        params = config.get("init_args", dict())
    elif "target" in config:
        target = config["target"]
        params = config.get("params", dict())
    return partial(get_obj_from_str(target), **params)


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


def make_path_absolute(path):
    fs, p = fsspec.core.url_to_fs(path)
    if fs.protocol == "file":
        return str(Path(p).absolute())
    return path


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


def instantiate_from_config(config):
    if "class_path" in config:
        # PyTorch Lightning syntax
        target = config["class_path"]
        params = config.get("init_args", dict())
    elif "target" in config:
        target = config["target"]
        params = config.get("params", dict())
    else:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(target)(**params)


def get_obj_from_str(string: str, reload: bool = False, invalidate_cache: bool = True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def append_zero(x: Tensor) -> Tensor:
    return torch.cat([x, x.new_zeros([1])])


def append_dims(x: Tensor, target_dims: int) -> Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def load_model_from_config(config, ckpt: PathLike, verbose=True, freeze=True) -> Module:
    logger.info(f"Loading model from {ckpt}")
    ckpt = Path(ckpt)

    if ckpt.suffix == ".ckpt":
        lightning_state = torch.load(ckpt, map_location="cpu")
        if "global_step" in lightning_state:
            logger.info(f"Global Step for {ckpt.name}: {lightning_state['global_step']}")
        state_dict = lightning_state["state_dict"]
    elif ckpt.suffix == ".safetensors":
        state_dict = load_safetensors(ckpt)
    else:
        raise ValueError(f"Unknown file extension {ckpt.suffix}")

    model: Module = instantiate_from_config(config.model)

    m, u = model.load_state_dict(state_dict, strict=False)

    if len(m) > 0 and verbose:
        logger.info("missing keys:")
        logger.info(m)
    if len(u) > 0 and verbose:
        logger.info("unexpected keys:")
        logger.info(u)

    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)

    return model


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
