import logging
from contextlib import contextmanager

import numpy as np

from neurosis import is_debug


def normalize8(x: np.ndarray) -> np.ndarray:
    min = x.min()
    max = x.max()

    x = ((x - min) / (max - min)) * 255
    x = x.round().astype(np.uint8)
    return x


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
