import logging
from io import BytesIO
from os import PathLike, getenv, getpid
from time import sleep
from typing import Optional

import pandas as pd
from botocore.exceptions import ConnectionError
from lightning.pytorch import LightningDataModule
from PIL import Image
from pymongo import MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongoarrow.api import find_pandas_all
from pymongoarrow.schema import Schema
from s3fs import S3FileSystem
from torch import Generator, Tensor
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from neurosis.dataset.base import NoBucketDataset
from neurosis.dataset.mongo.settings import MongoSettings, get_mongo_settings
from neurosis.dataset.utils import clear_fsspec, pil_crop_square, set_s3fs_opts
from neurosis.utils import maybe_collect
from neurosis.utils.image import pil_ensure_rgb

logger = logging.getLogger(__name__)


class MongoVAEDataset(NoBucketDataset):
    def __init__(
        self,
        settings: MongoSettings,
        resolution: int | tuple[int, int] = 256,
        batch_size: int = 1,
        image_key: str = "image",
        *,
        path_key: str = "s3_path",
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        s3_bucket: Optional[str] = None,
        s3fs_kwargs: dict = {},
        pma_schema: Optional[Schema] = None,
        retries: int = 3,
        retry_delay: int = 5,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(resolution, batch_size, **kwargs)
        self.pid = getpid()
        self.settings = settings

        self.image_key = image_key
        self.path_key = path_key
        self.resampling = resampling

        # load S3_ENDPOINT_URL from env if not already present
        if s3_endpoint_env := getenv("S3_ENDPOINT_URL", None):
            s3fs_kwargs.setdefault("endpoint_url", s3_endpoint_env)

        self.s3fs_kwargs = s3fs_kwargs
        self.s3_bucket = s3_bucket
        self.fs: S3FileSystem = None
        self.client: MongoClient = None
        self.pma_schema: Schema = pma_schema

        self.retries = retries
        self.retry_delay = retry_delay
        self.seed = seed

        # load meta
        logger.debug(
            f"Preloading dataset from mongodb://<host>/{self.settings.database}.{self.settings.collection}"
        )
        self._count: int = None
        self._first_getitem = True
        self._preload()

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return self._count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if self._first_getitem:
            logger.debug(f"First __getitem__ (idx {index}) - refreshing clients")
            self.refresh_clients()
            self._first_getitem = False

        sample: pd.Series = self.samples.iloc[index]
        image = self._get_image(sample[self.path_key])
        image, crop_coords = pil_crop_square(image, self.resolution, self.resampling)

        return {
            self.image_key: self.transforms(image),
            "crop_coords_top_left": crop_coords,
        }

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

    @property
    def collection(self) -> MongoCollection:
        return self.client.get_database(self.settings.db_name).get_collection(self.settings.coll_name)

    def _preload(self):
        self.refresh_clients()

        if self._count is None:
            logger.info(f"Counting documents in {self.settings.coll_name}")
            self._count = self.settings.count

        if not isinstance(self.samples, pd.DataFrame):
            logger.info(f"Loading metadata for {self._count} documents, this may take a while...")
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
            except ConnectionError:
                logger.exception(f"Caught connection error on {path}, retry in {self.retry_delay}s")
                sleep(self.retry_delay)
            attempts += 1

        if not isinstance(image, bytes):
            raise FileNotFoundError(f"Failed to load image from {path}")
        # load image and ensure RGB colorspace
        image = Image.open(BytesIO(image))
        image = pil_ensure_rgb(image)
        return image


class MongoVAEModule(LightningDataModule):
    def __init__(
        self,
        config_path: PathLike,
        resolution: int | tuple[int, int] = 256,
        batch_size: int = 1,
        image_key: str = "image",
        *,
        path_key: str = "s3_path",
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        s3_bucket: Optional[str] = None,
        s3fs_kwargs: dict = {},
        pma_schema: Optional[Schema] = None,
        seed: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.mongo_settings = get_mongo_settings(config_path)

        self.dataset = MongoVAEDataset(
            settings=self.mongo_settings,
            resolution=resolution,
            batch_size=batch_size,
            image_key=image_key,
            path_key=path_key,
            resampling=resampling,
            s3_bucket=s3_bucket,
            s3fs_kwargs=s3fs_kwargs,
            pma_schema=pma_schema,
        )
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

    @property
    def batch_size(self):
        return self.dataset.batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        logger.info(f"Refreshing dataset clients for {stage}")
        self.dataset.refresh_clients()

    def train_dataloader(self):
        if self.seed is not None:
            logger.info(f"Setting seed {self.seed} for train dataloader")
            generator = Generator().manual_seed(self.seed)
            sampler = RandomSampler(self.dataset, generator=generator)
        else:
            sampler = SequentialSampler(self.dataset)

        return DataLoader(
            self.dataset,
            batch_sampler=BatchSampler(sampler, self.batch_size, self.drop_last),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=mongo_worker_init,
            timeout=60.0,
        )


def mongo_worker_init(worker_id: int = -1):
    logger.debug(f"Worker {worker_id} initializing")
    clear_fsspec()
    set_s3fs_opts()
