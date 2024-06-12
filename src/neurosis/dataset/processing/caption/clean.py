from typing import Optional

import numpy as np
import pandas as pd
from numpy.random import permutation

from neurosis.dataset.utils import clean_word


def shuffle_tags(
    tags: list[str] | np.ndarray | pd.Series,
    keep: Optional[int] = None,
) -> list[str]:
    """
    Shuffle a tag list, optionally keeping some tags in place.

    Args:
        tags (list[str] | np.ndarray | pd.Series): List of tags to shuffle
        keep (Optional[int], optional): Number of tags to keep. Defaults to None.
                If positive, the first `keep` tags will be kept in place.
                If zero or None, all tags will be shuffled.
                If negative, all tags will be kept in place.

    Returns:
        list[str]: Shuffled tag list
    """
    if keep is None or keep == 0:  # shuffle all tags
        return permutation(tags).tolist()
    if 0 < keep < len(tags):  # shuffle [keep:] tags only
        return tags[:keep] + permutation(tags[keep:]).tolist()
    return tags  # keep all tags


def clean_tag_list(
    tags: list[str] | np.ndarray | pd.Series,
    word_sep: str = "_",
    shuffle: bool = False,
    keep: Optional[int] = None,
) -> list[str]:
    """
    Clean and (optionally) shuffle a list of tags.

    Cleaning involves decoding byte strings as UTF-8, normalizing separators,
    and removing leading/trailing whitespace.

    Args:
        tags (`list[str] | np.ndarray | pd.Series`): List of tags to clean
        word_sep (`str`, optional): Separator used for multi-word tags. Defaults to "_".
                    Underscores and spaces in tags will be replaced with this separator.
        shuffle (`bool`, optional): Whether to shuffle the list or not. Defaults to False.
        keep (`Optional[int]`, optional): See `shuffle_tags()`. Defaults to None.

    Returns:
        list[str]: Cleaned and (optionally) shuffled tag list
    """
    # clean each word
    tags = [clean_word(word_sep, x) for x in tags]
    # shuffle tags if needed
    if shuffle:
        tags = shuffle_tags(tags, keep=keep)
    return tags
