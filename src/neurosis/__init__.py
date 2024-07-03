try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from functools import lru_cache
from os import getenv
from pathlib import Path
from warnings import filterwarnings

import pandas as pd
from einops._torch_specific import allow_ops_in_compiled_graph
from PIL import Image
from rich.console import Console
from rich.traceback import install as _install_traceback

# make PIL be quiet about Big Images
Image.MAX_IMAGE_PIXELS = None
del Image

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent.parent

is_debug = getenv("NEUROSIS_DEBUG", None) is not None
if is_debug is True:
    is_debug = getenv("NEUROSIS_DEBUG").lower() not in ("0", "false", "no")

local_rank = int(getenv("LOCAL_RANK", "0"))

_ = _install_traceback(show_locals=is_debug, width=120, word_wrap=True)

# set up the console
console = Console(highlight=not is_debug)

if is_debug:
    console.log("[bold red]NEUROSIS_DEBUG[/bold red] is set")
    console.log("Enabling use of einops functions in TorchDynamo graphs...")

# enable einops to be used in torch.compile()
allow_ops_in_compiled_graph()
del allow_ops_in_compiled_graph

# enable Pandas copy-on-write
pd.options.mode.copy_on_write = True


@lru_cache(maxsize=4)
def get_dir(dirname: str = "data") -> Path:
    if PACKAGE_ROOT.name == "src":
        # we're installed in editable mode from within the repo
        dirpath = PACKAGE_ROOT.parent.joinpath(dirname)
    else:
        # we're installed normally, so we just use the current working directory
        dirpath = Path.cwd().joinpath(dirname)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath.absolute()


# shut
filterwarnings("ignore", category=FutureWarning, message="`Transformer2DModelOutput` is deprecated")
