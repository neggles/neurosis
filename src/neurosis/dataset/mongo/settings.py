import logging
from functools import cached_property
from hashlib import sha1
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
    limit: Optional[int] = Field(None)
    skip: Optional[int] = Field(None)

    @computed_field
    @property
    def kwargs(self) -> dict:
        args = {}
        if self.projection is not None:
            args.update({"projection": self.projection})
        if self.sort is not None:
            args.update({"sort": self.sort})
        if self.skip is not None:
            args.update({"skip": self.skip})
        if self.limit is not None:
            args.update({"limit": self.limit})
        return args


class MongoSettings(BaseSettings):
    uri: MongoDsn = Field(..., description="MongoDB URI")
    username: Optional[str] = Field(None, description="Username for the user")
    password: Optional[str] = Field(None, description="Password for the user")

    authMechanism: Optional[str] = Field("SCRAM-SHA-256", description="Authentication mechanism")
    authSource: Optional[str] = Field("admin", description="Database to authenticate against")
    tls: bool = Field(False, description="Use TLS")
    tlsInsecure: Optional[bool] = Field(True, description="Allow insecure TLS connections")

    db_name: str = Field(..., description="Database to query", alias="database")
    coll_name: str = Field(..., description="Collection to query", alias="collection")
    query: Query = Field(description="Query to run on the collection", default_factory=Query)

    caption_array: bool = Field(False, description="True if `caption` is an array of strings")

    model_config: SettingsConfigDict = dict(
        arbitrary_types_allowed=True,
        env_prefix="mongo_",
    )

    @computed_field
    @cached_property
    def query_dict(self) -> dict[str, Any]:
        qdict = dict(self.query)
        return qdict

    @computed_field
    @cached_property
    def client(self) -> MongoClient:
        mclient = self.new_client()
        return mclient

    @computed_field
    @cached_property
    def database(self) -> Database:
        db = self.client[self.db_name]
        return db

    @computed_field
    @cached_property
    def collection(self) -> Collection:
        coll = self.database[self.coll_name]
        return coll

    @computed_field
    @cached_property
    def count(self) -> int:
        logger.info(f"Counting documents in {self.db_name}/{self.coll_name} for query...")
        aggr = [
            {"$match": self.query.filter},
            {"$project": {"_id": 1}},
            {"$count": "count"},
        ]
        if self.query.sort is not None:
            aggr.insert(1, {"$sort": dict(self.query.sort)})
        if self.query.skip is not None:
            aggr.insert(-2, {"$skip": self.query.skip})
        if self.query.limit is not None:
            aggr.insert(-2, {"$limit": self.query.limit})
        count = self.collection.aggregate(aggr).next()["count"]
        return count

    @computed_field
    @cached_property
    def query_hash(self) -> str:
        q_json = self.query.model_dump_json().encode("utf-8")
        return sha1(q_json).hexdigest().lower()

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


def get_mongo_settings(path: PathLike = DEFAULT_MONGO_CONFIG) -> MongoSettings:
    path = Path(path).resolve()
    if path.exists() and path.is_file():
        logger.info(f"Loading Mongo config from {path}")
        return MongoSettings.model_validate_json(path.read_bytes(), strict=True)
    else:
        logger.warning(f"Mongo config file {path} does not exist, using env")
        return MongoSettings()
