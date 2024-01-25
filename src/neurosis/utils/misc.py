import logging
from contextlib import contextmanager
from itertools import islice
from typing import Generator, Iterable, Iterator, List, Optional, TypeVar

import numpy as np

from neurosis import is_debug

T = TypeVar("T")


# https://github.com/python/cpython/issues/98363
def batched(iterable: Iterable[T], n: int) -> Generator[List[T], None, None]:
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be >= 1")
    it: Iterator[T] = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def ndimage_to_f32(x: np.ndarray, zero_min: Optional[bool] = None) -> np.ndarray:
    zero_min = x.min() >= 0 if zero_min is None else zero_min

    if zero_min:
        x = x / 255.0  # 0-255 -> 0.0-1.0
    else:
        x = (x / 127.5) - 1.0  # 0-255 -> -1.0-1.0

    x = x.clip(min=-1.0, max=1.0)

    return x.astype(np.float32)


def ndimage_to_u8(x: np.ndarray, zero_min: Optional[bool] = None) -> np.ndarray:
    zero_min = x.min() >= 0 if zero_min is None else zero_min

    if zero_min is True:
        x = x * 255.0  # 0 to 1 -> 0 to 255
    elif zero_min is False:
        x = (x * 127.5) + 127.5  # -1 to +1 -> 0 to 255

    x = x.clip(min=0.0, max=255.0)

    return x.round().astype(np.uint8)


def ndimage_to_u8_norm(x: np.ndarray) -> np.ndarray:
    min = x.min()
    max = x.max()

    x = ((x - min) / (max - min)) * 255
    return x.round().astype(np.uint8)


class HFLoadFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg.startswith("Some weights of the model checkpoint"):
            return False
        return True


@contextmanager
def silence_hf_load_warnings():
    # get root, transformers, and diffusers loggers\
    root_logger = logging.getLogger()
    tfrs_logger = logging.getLogger("transformers.modeling_utils")
    dfrs_logger = logging.getLogger("diffusers.modeling_utils")

    # create a filter obj
    hf_filter = HFLoadFilter()

    # don't silence if we're at debug level
    if is_debug is False:
        root_logger.addFilter(hf_filter)
        tfrs_logger.addFilter(hf_filter)
        dfrs_logger.addFilter(hf_filter)
        try:
            # do the thing
            yield
        finally:
            # remove filters
            root_logger.removeFilter(hf_filter)
            tfrs_logger.removeFilter(hf_filter)
            dfrs_logger.removeFilter(hf_filter)
    else:
        try:
            yield
        finally:
            pass
