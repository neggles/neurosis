import numpy as np


def np_text_decode(text: np.bytes_ | str | list, aslist: bool = False):
    if not isinstance(text, list):
        text = [text]
    text = [x.decode("utf-8") if isinstance(x, np.bytes_) else x for x in text]
    if len(text) == 1 and not aslist:
        text = text[0]
    return text
