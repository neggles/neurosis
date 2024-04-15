import gc
from pathlib import Path

import psutil


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
