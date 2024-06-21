import gc
from os import getenv
from pathlib import Path
from socket import gethostname
from typing import Optional

import psutil
from torch import distributed as dist


def maybe_collect(threshold: float = 75.0):
    """
    Call python GC if memory usage is greater than <threshold>% of total available memory.
    This is used to deal with Ray not triggering global garbage collection due to how GC cycles are tracked.
    """
    if psutil.virtual_memory().percent >= threshold:
        gc.collect()
    return


def get_next_dir(path: Path, prefix: str, sep: str = "_", offset: int = 0) -> Path:
    if len(sep) < 1:
        raise ValueError("Separator must have at least one character, got empty string")

    subdirs = [x.name for x in path.iterdir() if x.is_dir() and x.name.startswith(prefix + sep)]
    if not subdirs:
        return path.joinpath(f"{prefix}{sep}0")

    newest = sorted(subdirs, key=lambda x: x.rsplit(sep, 1)[-1], reverse=True)[0]
    idx = int(newest.rsplit(sep, 1)[-1]) + 1

    while path.joinpath(f"{prefix}{sep}{idx}").exists():
        # this shouldn't be necessary but covers against races or there being a file with the same name
        idx += 1

    # subtract offset (to account for rank in distributed training, for example)
    idx -= offset

    return path.joinpath(f"{prefix}{sep}{idx}")


def get_node_name() -> str:
    node_name = getenv("SLURMD_NODENAME")
    if not node_name:
        node_name = gethostname()
    return node_name


def get_rank() -> Optional[int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return None


def get_rank_str(prefix: str = "") -> Optional[str]:
    if dist.is_available() and dist.is_initialized():
        n_ranks = dist.get_world_size()
        rank = dist.get_rank()
        match n_ranks:
            case _ if n_ranks > 999:
                return f"{prefix}{rank:04d}"
            case _ if n_ranks > 99:
                return f"{prefix}{rank:03d}"
            case _ if n_ranks > 9:
                return f"{prefix}{rank:02d}"
            case _:
                return f"{prefix}{rank}"
    return None


def prepend_node_name(s: str, sep: str = "") -> str:
    if node_name := get_node_name():
        return f"{node_name}{sep}{s}"
    return s


def prepend_rank(s: str, sep: str = "-") -> str:
    if rank := get_rank_str():
        return f"r{rank}{sep}{s}"
    return s
