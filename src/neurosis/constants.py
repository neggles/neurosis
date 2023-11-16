from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

IMAGE_EXTNS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]

CHECKPOINT_EXTNS = [".pt", ".pth", ".ckpt", ".safetensors"]

TEST_DS_PATH = PACKAGE_ROOT.parent.parent.joinpath("data/ine")

# misc size stuff
MBYTE = 2**20
GBYTE = 2**30
