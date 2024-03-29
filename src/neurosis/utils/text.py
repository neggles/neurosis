import numpy as np


def np_text_decode(text: np.bytes_ | str | list, aslist: bool = False):
    if not isinstance(text, list):
        text = [text]
    # unwrap any 0-dim arrays
    text = [x.tobytes() if isinstance(x, np.ndarray) else x for x in text]
    # decode bytes to string
    text = [x.decode("utf-8") if isinstance(x, (np.bytes_, bytes)) else x for x in text]
    if len(text) == 1 and not aslist:
        text = text[0]
    return text
