from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from neurosis import get_dir

DEFAULT_QUERY_CONFIG = get_dir("configs").joinpath("mongo", "default-query.json")


class MongoQuery(BaseModel):
    filter: dict[str, Any] = Field(default_factory=dict)
    projection: Optional[dict[str, Any]] = Field(None)
    sort: Optional[list[tuple[str, int]]] = Field(None)


class QueryConfig(BaseModel):
    queries: list[MongoQuery] = Field(default_factory=list)
    gridfs_prefix: str = Field(default="fs")


@lru_cache(maxsize=4)
def get_query_config(path: PathLike = DEFAULT_QUERY_CONFIG) -> QueryConfig:
    path = Path(path)
    return QueryConfig.parse_file(path)
