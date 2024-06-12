from datetime import datetime
from typing import Optional


def booru_score_tag(score_up: int, score_down: int = 0) -> Optional[str]:
    """Map the up/down score of an image to a quality tag."""
    score = score_up - score_down
    match score:
        case x if x >= 150:
            tag = "masterpiece"
        case x if x >= 100:
            tag = "best quality"
        case x if x >= 75:
            tag = "high quality"
        case x if x >= 25:
            tag = "medium quality"
        case x if x >= 0:
            tag = "normal quality"
        case x if x >= -5:
            tag = "low quality"
        case x if x < -5:
            tag = "worst quality"
        case _:
            tag = None
    return tag


def source_tag(source: str) -> str:
    source = source.lower()
    if source.startswith("danbooru") or "gwern" in source:
        return "danbooru"
    return source


def age_tag(created_at: datetime | str, auto_range: bool = False) -> str:
    ref_year = datetime.now().year if auto_range else 2025
    if not isinstance(created_at, datetime):
        created_at = datetime.fromisoformat(created_at)

    match created_at.year:
        case x if x < ref_year - 15:
            return "oldest"
        case x if x < ref_year - 10:
            return "old"
        case x if x < ref_year - 5:
            return "new"
        case _:
            return "newest"
