from contextlib import contextmanager
from importlib import resources
from importlib.abc import Traversable
from io import BufferedReader
from typing import Any, Generator


@contextmanager
def package_file(dir: str, name: str) -> Generator[Traversable, Any, None]:
    file = resources.files(f"{__package__}.{dir}").joinpath(name)
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found in {resources.files(__package__)}")
    try:
        yield file
    finally:
        pass


@contextmanager
def lpips_checkpoint(name: str = "vgg_lpips") -> Generator[BufferedReader, Any, None]:
    lpips_file = resources.files(f"{__package__}.lpips").joinpath(f"{name}.pth")
    if not lpips_file.exists():
        raise FileNotFoundError(f"File {lpips_file} not found in {__package__}.lpips")
    try:
        yield lpips_file.open("rb")
    finally:
        pass
