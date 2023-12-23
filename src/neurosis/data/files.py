from contextlib import contextmanager
from importlib import resources
from importlib.abc import Traversable
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
def open_package_file(dir: str, name: str, mode: str = "r") -> Generator[Traversable, Any, None]:
    file = resources.files(f"{__package__}.{dir}").joinpath(name)
    if not file.exists():
        raise FileNotFoundError(f"File {file} not found in {resources.files(__package__)}")
    try:
        yield file.open(mode)
    finally:
        pass
