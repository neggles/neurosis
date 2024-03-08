import logging
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Rating(str, Enum):
    """String enumerator for ratings following booru-style classification convention"""

    G = "general"
    S = "sensitive"
    Q = "questionable"
    E = "explicit"


def how_lewd_is_this(
    scores: dict[str | Rating, float],
    src_rating: Optional[Rating] = None,
    sfw_tag: Optional[str] = None,
    nsfw_tag: Optional[str] = "nsfw",
    source_confidence: float = 0.75,
) -> tuple[Optional[str], Rating]:
    """
    Determines the NSFW-ness of an image based on the provided scores and optional rating from the source.

    This function maps from rating probabilities (provided by a classifier) to a tag.
    If the datasource provides a rating, its score is bumped up to at least the source confidence.
    The function then determines the likelihood of the image being NSFW based on the highest score.

    The booru "sensitive" rating is somewhat ambiguous, containing images that (depending on context)
    may qualify as either NSFW or SFW. If the highest score falls into this category, the "tie" (so to
    speak) is broken by comparing the score of the "General" rating to the maximum of the "Questionable"
    and "Explicit" ratings. If the "General" score is higher, the image is considered SFW; otherwise, it
    is considered NSFW.

    If the rating is unknown, the function logs a warning and assumes the image is SFW.

    Parameters:
    scores (dict[Rating, float]): A dictionary mapping ratings to their respective scores.
    src_rating (Optional[Rating]): The rating from the source, if available. Defaults to None. `Rating` is a string enum.
    sfw_tag (Optional[str]): The tag to return for Safe For Work (SFW) images. Defaults to `None` (no tag).
    nsfw_tag (Optional[str]): The tag to return for Not Safe For Work (NSFW) images. Defaults to "nsfw".
    source_confidence (float): The assumed confidence of the source's rating info. Defaults to 0.75.

    Returns a tuple of:
    Optional[str]: Either `sfw_tag` or `nsfw_tag` depending on the determined NSFW-ness of the image.
    Rating: the derived overall rating of the image.
    """
    if src_rating is not None:
        # if we have a rating from the source, bump up its score to at least source_confidence
        scores[src_rating] = max(scores[src_rating], source_confidence)

    # determine the highest score and convert it to a `Rating`
    rating = max(scores, key=scores.get)
    rating = Rating(rating)

    match rating:
        case Rating.G:
            return sfw_tag, rating
        case Rating.S:
            probably_sfw = scores[Rating.G] > max(scores[Rating.Q], scores[Rating.E])
            if probably_sfw:
                return sfw_tag, rating
            else:
                return nsfw_tag, rating
        case [Rating.Q, Rating.E]:
            return nsfw_tag, rating
        case _:
            logger.warning(f"Got unknown rating '{rating}', assuming SFW...")
            return sfw_tag, rating


def make_loli_great_again(tags: pd.Series, rating: Optional[Rating] = None):
    """
    The goal of this function is to remove the NSFW implications of the 'loli' and 'shota' tags.

    These two words (in an anime context) merely mean "young girl" or "young boy", and primarily
    refer to a specific character design style (smol) which has no inherent NSFW connotations.

    Unfortunately, the majority of our dataset comes from sites with very similar policies around
    image tagging; these shared rules define 'loli' and 'shota' as carrying an implication of
    explicit content and instead use "female child" or "male child" for non-explicit imagery.

    We consider this a gross injustice against those who merely wish to generate images along the
    lines of characters such as Beatrice from Re:Zero, etc. We also find "female child" and
    "male child" to be kind of gross, tbh.

    To be clear, all explicit content featuring these tags has been removed from our dataset. (we
    just really enjoy not going to jail, you know?) so if things are left as-is we get a model
    that doesn't know what loli is, and responds to icky "<blank> child" tags.

    Thus, we must Make Loli Great Again.

    This function remaps all instances of "<blank> child" to 'loli' or 'shota' as appropriate,
    and drops the 'female child' and 'male child' tags entirely, collapsing them to just "child".

    We believe this will result in a model that is significantly less prone to generating explicit
    content when users provide what they consider non-explicit prompts.

    On the other hand since these tags occur in less than 1% of dataset samples it might do nothing.

    Parameters:
    tags (pandas.Series): a Pandas `series` object containing a sample's tag list.
    rating (Rating): a Rating enum or string indicating the image's inferred rating

    Returns:
    pandas.Series: The modified tag list
    """

    tag_list = list(tags)
    modified = False

    if "female_child" in tag_list or all((x in tag_list for x in ["child", "1girl"])):
        modified = True
        tag_list.remove("female_child")
        if "loli" not in tag_list:
            tag_list.append("loli")

    if "male_child" in tag_list or all((x in tag_list for x in ["child", "1boy"])):
        modified = True
        tag_list.remove("male_child")
        if "shota" not in tag_list:
            tag_list.append("shota")

    if modified and "child" not in tag_list:
        tag_list.append("child")

    if list(tags) != tag_list:
        tags = pd.Series(set(tag_list), dtype=tags.dtype)
    return tags
