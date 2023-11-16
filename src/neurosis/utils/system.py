import gc

import psutil


def maybe_collect(threshold: float = 75.0):
    """
    Call python GC if memory usage is greater than <threshold>% of total available memory.
    This is used to deal with Ray not triggering global garbage collection due to how GC cycles are tracked.
    """
    if psutil.virtual_memory().percent >= threshold:
        gc.collect()
    return
