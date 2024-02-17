from contextlib import contextmanager
from functools import lru_cache
from importlib import resources
from io import BufferedReader
from typing import Any, Generator

from PIL import ImageFont
from safetensors.torch import load_file as safetensors_load
from torch import load as torch_load

from .files import open_package_file


@lru_cache(maxsize=8)
def get_image_font(name: str = "NotoSansMono", size: int = 10) -> ImageFont.FreeTypeFont:
    with open_package_file("fonts", f"{name}.ttf", "rb") as f:
        font = ImageFont.truetype(f, size=size)
    return font


@contextmanager
def lpips_checkpoint(name: str = "vgg", version: str = "v0.1") -> Generator[BufferedReader, Any, None]:
    ckpt_name = f"{name}_lpips_{version}"

    rsrc_dir = resources.files(f"{__package__}.lpips")
    try:
        if rsrc_dir.joinpath(f"{ckpt_name}.safetensors").exists():
            yield safetensors_load(rsrc_dir.joinpath(f"{ckpt_name}.safetensors"))
        elif rsrc_dir.joinpath(f"{ckpt_name}.pth").exists():
            yield torch_load(rsrc_dir.joinpath(f"{ckpt_name}.pth"), map_location="cpu", weights_only=True)
    finally:
        pass
