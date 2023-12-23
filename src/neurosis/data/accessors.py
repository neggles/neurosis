from contextlib import contextmanager
from functools import lru_cache
from importlib import resources
from io import BufferedReader
from typing import Any, Generator

from PIL import ImageFont

from .files import open_package_file


@lru_cache(maxsize=2)
def get_image_font(name: str = "NotoSansMono", size: int = 10) -> ImageFont.FreeTypeFont:
    with open_package_file("fonts", f"{name}.ttf", "rb") as f:
        font = ImageFont.truetype(f, size=size)
    return font


@contextmanager
def lpips_checkpoint(name: str = "vgg_lpips") -> Generator[BufferedReader, Any, None]:
    lpips_file = resources.files(f"{__package__}.lpips").joinpath(f"{name}.pth")
    if not lpips_file.exists():
        raise FileNotFoundError(f"File {lpips_file} not found in {__package__}.lpips")
    try:
        yield lpips_file.open("rb")
    finally:
        pass
