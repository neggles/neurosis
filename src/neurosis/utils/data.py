from contextlib import contextmanager
from importlib import resources
from pathlib import Path


MODEL_EXTNS = [".pt", ".ckpt", ".pth", ".safetensors"]


def get_data_path(file: str = ..., module: str = f"{__name__.split('.')[0]}.data") -> Path:
    """Get the path to a module data file"""
    module_data = resources.files(module)
    file_path = module_data.joinpath(file)

    if not module_data.joinpath(file).exists():
        raise FileNotFoundError(f"File {file} not found in {module_data}")

    return file_path


@contextmanager
def package_data_file(name: str, package: str = __package__, mode: str = "rb") -> str:
    package_files = resources.files(package)
    target_file = package_files.joinpath(name)
    if not target_file.exists():
        for ext in MODEL_EXTNS:
            target_file = package_files.joinpath(f"{name}{ext}")
            if target_file.exists():
                break

    if not target_file.exists():
        raise FileNotFoundError(f"Could not find file {name} in package {package}")
    try:
        yield target_file.open(mode)
    finally:
        pass
