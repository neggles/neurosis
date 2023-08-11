from datetime import datetime
from typing import Optional

from bson import ObjectId
from pydantic import BaseModel, Field
from pydantic_mongo import AbstractRepository, ObjectIdField
from pymongo import MongoClient


def current_unix_time() -> int:
    return int(datetime.now().timestamp())


class KImageMetadata(BaseModel):
    id: ObjectIdField = Field(default_factory=ObjectId, alias="_id")
    post_id: int = Field(..., allow_mutation=False)
    post_date: int = Field(..., allow_mutation=False)
    scrape_date: int = Field(default_factory=current_unix_time)
    scraped_by: Optional[str] = Field(None)

    width: Optional[int] = Field(None)
    height: Optional[int] = Field(None)
    byte_size: Optional[int] = Field(None, ge=0)
    hash: Optional[str] = Field(None, min_length=32, max_length=128)

    favs: Optional[int] = Field(None)
    score: Optional[int] = Field(None)
    tags: list[str] = Field(default_factory=list)

    s3_url: Optional[str] = Field(None)

    class Config:
        json_encoders = {
            ObjectId: str,  # BSON Object ID value needs explicit conversion to str
        }


class KRepository(AbstractRepository[KImageMetadata]):
    class Meta:
        collection_name = "images_meta"
