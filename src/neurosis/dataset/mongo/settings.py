from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs

from gridfs import GridFS
from pydantic import BaseModel, Field, MongoDsn, Protocol
from pymongo import MongoClient
from pymongo.database import Database

from neurosis import get_dir

DEFAULT_MONGO_CONFIG = get_dir("configs").joinpath("mongo", "default.json")


class Query(BaseModel):
    filter: dict[str, Any] = Field(default_factory=dict)
    projection: Optional[dict[str, Any]] = Field(None)
    sort: Optional[list[tuple[str, int]]] = Field(None)


class MongoSettings(BaseModel):
    uri: MongoDsn = Field(..., env="MONGO_URI")
    username: Optional[str] = Field(None)
    password: Optional[str] = Field(None)
    database: str = Field(..., description="Database to pull from", env="MONGO_DATABASE")
    collection: str = Field(..., description="Collection to pull from", env="MONGO_COLLECTION")
    gridfs_prefix: str = Field(default="fs", description="GridFS prefix", env="MONGO_GRIDFS_PREFIX")
    queries: list[Query] = Field(default_factory=list)

    _client: Optional[MongoClient] = Field(None, allow_mutation=True, init=False, repr=False)

    def get_client(self, new: bool = False) -> MongoClient:
        if not new and self._client is not None:
            return self._client

        # parse the query string for extra kwargs to pass to MongoClient
        mongo_kwargs: dict[str, Any] = parse_qs(self.uri.query)
        user = self.username or self.uri.user
        password = self.password or self.uri.password
        # apply some sane defaults for authentication mode
        mongo_kwargs.setdefault("authSource", "admin")
        mongo_kwargs.setdefault("authMechanism", "SCRAM-SHA-256")

        client = MongoClient(
            host=self.uri.host, port=self.uri.port, username=user, password=password, **mongo_kwargs
        )
        if new:
            # just return the new client
            return client
        else:
            # save the client for future calls and return it
            self._client = client
            return self._client

    def get_database(self) -> Database:
        client = self.get_client()
        return client[self.database]

    def get_fsclient(self) -> GridFS:
        return GridFS(self.get_database(), self.gridfs_prefix)


@lru_cache(maxsize=4)
def get_mongo_settings(path: PathLike = DEFAULT_MONGO_CONFIG) -> MongoSettings:
    path = Path(path)
    return MongoSettings.parse_file(path, proto=Protocol.json)
