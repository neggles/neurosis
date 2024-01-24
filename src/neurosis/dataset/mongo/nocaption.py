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
from torch.utils.data import DataLoader, random_split

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
        """Helper func to replace the current clients with new ones"""
        self.client = self.settings.new_client()

        # detect forks and reset fsspec
        pid = getpid()
        if self.fs is None or self.fs._pid != pid:
            logger.info(f"loader PID {pid} detected fork, resetting fsspec clients")
            import fsspec

            fsspec.asyn.reset_lock()
            self.fs = S3FileSystem(**self.s3fs_kwargs, skip_instance_cache=True)
            self.pid = pid

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
        train_split: float = 1.0,
        seed: int = 42,
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
        self.train_split = train_split
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

        # set in prepare_data
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        if self.train_split >= 1.0:
            logger.info("Train split is >=100%, not splitting dataset (validation set will be empty)")
            self.train_dataset = self.dataset
            return

        logger.info(f"Splitting dataset into train and val sets at {self.train_split} ratio")
        nsamples = len(self.dataset)
        train_samples = int(nsamples * self.train_split // self.dataset.batch_size * self.dataset.batch_size)

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_samples, nsamples - train_samples], Generator().manual_seed(self.seed)
        )
        logger.info(f"Train set: {len(self.train_dataset)} samples, Val set: {len(self.val_dataset)} samples")

    def setup(self, stage: str):
        logger.info(f"Refreshing dataset clients for {stage}")
        self.dataset.refresh_clients()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset if self.train_dataset is not None else self.dataset,
            batch_size=self.dataset.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=mongo_worker_init,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("No validation dataset available!")
        return DataLoader(
            self.val_dataset,
            batch_size=self.dataset.batch_size,
            num_workers=max(2, self.num_workers // 2),
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=mongo_worker_init,
        )


def mongo_worker_init(worker_id: int = -1):
    logger.info(f"Worker {worker_id} initializing")
    clear_fsspec()
    set_s3fs_opts()
