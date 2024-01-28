import logging
from os import PathLike
from typing import Literal, Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image
from pymongoarrow.schema import Schema
from torch import Tensor
from torch.utils.data import DataLoader

from neurosis.dataset.base import NoBucketDataset
from neurosis.dataset.utils import pil_crop_square

from .base import BaseMongoDataset, mongo_worker_init
from .settings import MongoSettings, get_mongo_settings

logger = logging.getLogger(__name__)


class MongoVAEDataset(BaseMongoDataset, NoBucketDataset):
    def __init__(
        self,
        settings: MongoSettings,
        *,
        image_key: str = "image",
        # passed to NoBucketDataset
        resolution: int | tuple[int, int] = 256,
        # passed to BaseMongoDataset along with settings
        batch_size: int = 1,
        path_key: str = "s3_path",
        extra_keys: list[str] | Literal["all"] = [],
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        s3_bucket: Optional[str] = None,
        s3fs_kwargs: dict = {},
        pma_schema: Optional[Schema] = None,
        retries: int = 3,
        retry_delay: int = 5,
        **kwargs,
    ):
        self.image_key = image_key
        self.batch_keys: list[str] = [image_key]

        BaseMongoDataset.__init__(
            self,
            settings=settings,
            batch_size=batch_size,
            path_key=path_key,
            extra_keys=extra_keys,
            resampling=resampling,
            s3_bucket=s3_bucket,
            s3fs_kwargs=s3fs_kwargs,
            pma_schema=pma_schema,
            retries=retries,
            retry_delay=retry_delay,
            **kwargs,
        )
        NoBucketDataset.__init__(
            self,
            resolution=resolution,
            **kwargs,
        )

        self.preload()

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
            "crop_coords_top_left": torch.tensor(crop_coords, dtype=torch.int32),
            **{k: torch.tensor(sample.get(k)) for k in self.extra_keys if k in sample},
        }


class MongoVAEModule(LightningDataModule):
    def __init__(
        self,
        config_path: PathLike,
        resolution: int | tuple[int, int] = 256,
        batch_size: int = 1,
        image_key: str = "image",
        *,
        path_key: str = "s3_path",
        extra_keys: list[str] | Literal["all"] = [],
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
            extra_keys=extra_keys,
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
        logger.debug(f"Refreshing dataset clients for {stage}")
        self.dataset.refresh_clients()

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            worker_init_fn=mongo_worker_init,
            timeout=60.0,
        )
