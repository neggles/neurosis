import logging
from functools import cached_property, lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, MongoDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from neurosis import get_dir

DEFAULT_MONGO_CONFIG = get_dir("configs").joinpath("mongo", "default.json")
logger = logging.getLogger(__name__)


class Query(BaseModel):
    filter: dict[str, Any] = Field(default_factory=dict)
    projection: Optional[dict[str, Any]] = Field(None)
    sort: Optional[list[tuple[str, int]]] = Field(None)


class MongoSettings(BaseSettings):
    uri: MongoDsn = Field(..., description="MongoDB URI")
    username: Optional[str] = Field(None, description="Username for the user")
    password: Optional[str] = Field(None, description="Password for the user")

    authMechanism: Optional[str] = Field(None, description="Authentication mechanism")
    authSource: Optional[str] = Field(None, description="Database to authenticate against")
    tls: bool = Field(False, description="Use TLS")
    tlsInsecure: Optional[bool] = Field(None, description="Allow insecure TLS connections")

    database_name: str = Field(..., description="Database to query", alias="database")
    collection_name: str = Field(..., description="Collection to query", alias="collection")
    query: Query = Field(description="Query to run on the collection")

    model_config: SettingsConfigDict = dict(
        arbitrary_types_allowed=True,
        env_prefix="mongo_",
    )

    @computed_field
    @cached_property
    def client(self) -> MongoClient:
        client = self.new_client()
        return client

    @computed_field
    @cached_property
    def database(self) -> Database:
        db = self.client[self.database_name]
        return db

    @computed_field
    @cached_property
    def collection(self) -> Collection:
        coll = self.database[self.collection_name]
        return coll

    @computed_field
    @cached_property
    def count(self) -> int:
        aggr = [
            {"$match": self.query.filter},
            {"$project": {"_id": 1}},
            {"$count": "count"},
        ]
        if self.query.sort is not None:
            aggr.insert(1, {"$sort": dict(self.query.sort)})
        count = self.collection.aggregate(aggr).next()["count"]
        return count

    def new_client(self) -> MongoClient:
        # parse the query string for extra kwargs to pass to MongoClient
        mongo_kwargs: dict[str, Any] = dict(self.uri.query_params())

        # apply some sane defaults for authentication mode
        if self.tls is not None:
            mongo_kwargs.setdefault("tls", self.tls)
        if self.tlsInsecure is not None:
            mongo_kwargs.setdefault("tlsInsecure", self.tlsInsecure)
        mongo_kwargs.setdefault("authSource", self.authSource)
        mongo_kwargs.setdefault("authMechanism", self.authMechanism)

        return MongoClient(
            host=self.uri.unicode_string(),
            username=self.username,
            password=self.password,
            **mongo_kwargs,
        )


@lru_cache(maxsize=4)
def get_mongo_settings(path: PathLike = DEFAULT_MONGO_CONFIG) -> MongoSettings:
    path = Path(path)
    if path.exists() and path.is_file:
        return MongoSettings.model_validate_json(path.read_bytes(), strict=True)
    else:
        logger.info(f"Mongo config file {path} does not exist, using env")
        return MongoSettings()
