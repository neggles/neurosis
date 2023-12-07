import numpy as np


def np_text_decode(text: np.bytes_ | str | list):
    if not isinstance(text, list):
        text = [text]
    text = [x.decode("utf-8") if isinstance(x, np.bytes_) else x for x in text]
    return text[0] if len(text) == 1 else text
