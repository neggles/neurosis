import numpy as np


def np_text_decode(text: np.bytes_ | str | list):
    if not isinstance(text, list):
        text = [text]
    text = [x.decode("utf-8") for x in text if isinstance(x, np.bytes_)]
    return text[0] if len(text) == 1 else text
