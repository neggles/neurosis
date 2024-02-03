import logging
from abc import abstractmethod
from io import BytesIO
from os import getenv, getpid
from time import sleep
from typing import Literal, Optional

import pandas as pd
from PIL import Image
from pymongo import MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongoarrow.api import find_pandas_all
from pymongoarrow.schema import Schema
from s3fs import S3FileSystem
from torch import Tensor
from torch.utils.data import Dataset

from neurosis.dataset.mongo.settings import MongoSettings
from neurosis.dataset.utils import clear_fsspec, set_s3fs_opts
from neurosis.utils import maybe_collect
from neurosis.utils.image import pil_ensure_rgb

logger = logging.getLogger(__name__)


def mongo_worker_init(worker_id: int = -1):
    logger.debug(f"Worker {worker_id} initializing")
    clear_fsspec()
    set_s3fs_opts()


class BaseMongoDataset(Dataset):
    def __init__(
        self,
        settings: MongoSettings,
        batch_size: int = 1,
        path_key: str = "s3_path",
        extra_keys: list[str] | Literal["all"] = [],
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        no_resize: bool = False,
        s3_bucket: Optional[str] = None,
        s3fs_kwargs: dict = {},
        pma_schema: Optional[Schema] = None,
        retries: int = 3,
        retry_delay: int = 5,
        **kwargs,
    ):
        self.pid = getpid()
        self.settings = settings
        self.batch_size = batch_size
        self.path_key = path_key
        self.resampling = resampling
        self.no_resize = no_resize

        # used to trigger a refresh check on first get just to be triple sure
        self._first_getitem = True

        # this is used to load extra metadata from the database
        if not hasattr(self, "batch_keys"):
            self.batch_keys: list[str] = []
        if path_key not in self.batch_keys:
            self.batch_keys.append(path_key)

        # if extra_keys is "all", then load all keys that are not consumed by the batch
        if isinstance(extra_keys, str) and extra_keys == "all":
            self.extra_keys = [x for x in self.query_keys if x not in self.batch_keys]
        else:
            self.extra_keys = extra_keys

        # set up for s3fs, load S3_ENDPOINT_URL from env if not already present
        if s3_endpoint_env := getenv("S3_ENDPOINT_URL", None):
            s3fs_kwargs.setdefault("endpoint_url", s3_endpoint_env)
        self.s3fs_kwargs = s3fs_kwargs
        self.s3_bucket = s3_bucket
        self.fs: S3FileSystem = None
        self.retries = retries
        self.retry_delay = retry_delay

        # set up for mongo
        self.client: MongoClient = None
        self.pma_schema: Schema = pma_schema

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return self.query_count

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        raise NotImplementedError("Abstract base class was called ;_;")

    @property
    def database(self) -> MongoCollection:
        return self.client.get_database(self.settings.db_name)

    @property
    def collection(self) -> MongoCollection:
        return self.database.get_collection(self.settings.coll_name)

    @property
    def query_keys(self) -> list[str]:
        return [k for k, v in self.settings.query.projection.items() if v not in [-1, 0]]

    @property
    def query_count(self):
        return self.settings.count

    def refresh_clients(self):
        """Helper func to replace the current clients with new ones post-fork etc."""
        pid = getpid()
        if self.client is None or self.pid != pid:
            self.client = self.settings.new_client()
            self.pid = pid

        if self.fs is None or self.fs._pid != pid:
            logger.debug(f"Loader detected fork, new PID {pid} - resetting fsspec clients")
            import fsspec

            fsspec.asyn.reset_lock()
            self.fs = S3FileSystem(**self.s3fs_kwargs, skip_instance_cache=True)

    def preload(self):
        self.refresh_clients()

        if not isinstance(self.samples, pd.DataFrame) or len(self.samples) == 0:
            logger.info(f"Loading metadata for {self.query_count} documents, this may take a while...")
            self.samples: pd.DataFrame = find_pandas_all(
                self.collection,
                query=dict(self.settings.query.filter),
                schema=self.pma_schema,
                **self.settings.query.kwargs,
            )

        logger.debug("Preload complete!")
        maybe_collect()

    def _get_image(self, path: str) -> Image.Image:
        if self.fs is None:
            self.refresh_clients()

        # prepend bucket if not already present
        image = None
        path = f"{self.s3_bucket}/{path}" if self.s3_bucket is not None else path

        # get image
        attempts = 0
        while attempts < self.retries and image is None:
            try:
                image = self.fs.cat(path)
            except Exception as e:
                logger.exception(f"Caught connection error on {path}, retry in {self.retry_delay}s")
                sleep(self.retry_delay + attempts)
                attempts += 1
                if attempts >= self.retries:
                    raise FileNotFoundError(f"Failed to load image from {path}") from e

        if not isinstance(image, bytes):
            raise FileNotFoundError(f"Failed to load image from {path}")
        # load image and ensure RGB colorspace
        image = Image.open(BytesIO(image))
        image = pil_ensure_rgb(image)
        return image
