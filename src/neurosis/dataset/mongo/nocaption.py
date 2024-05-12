import logging
from os import PathLike
from typing import Literal, Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image
from pymongoarrow.schema import Schema
from torch.utils.data import DataLoader

from neurosis.dataset.base import FilesystemType, NoBucketDataset, SampleType
from neurosis.dataset.mongo.base import BaseMongoDataset, mongo_worker_init
from neurosis.dataset.mongo.settings import MongoSettings, get_mongo_settings
from neurosis.dataset.processing.transform import DataTransform
from neurosis.dataset.utils import pil_crop_random, pil_crop_square

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
        reverse: bool = False,
        shuffle: bool = False,
        data_transforms: list[DataTransform] = [],
        no_resize: bool = False,
        fs_type: str | FilesystemType = "s3",
        path_prefix: Optional[str] = None,
        fsspec_kwargs: dict = {},
        pma_schema: Optional[Schema] = None,
        retries: int = 3,
        retry_delay: int = 5,
        skip_preload: bool = False,
        **kwargs,
    ):
        self.image_key = image_key
        self.batch_keys: list[str] = [
            image_key,
            "crop_coords_top_left",
        ]

        BaseMongoDataset.__init__(
            self,
            settings=settings,
            batch_size=batch_size,
            path_key=path_key,
            extra_keys=extra_keys,
            resampling=resampling,
            reverse=reverse,
            shuffle=shuffle,
            data_transforms=data_transforms,
            no_resize=no_resize,
            fs_type=fs_type,
            path_prefix=path_prefix,
            fsspec_kwargs=fsspec_kwargs,
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

        if self.no_resize:
            self.img_load_fn = pil_crop_random
        else:
            self.img_load_fn = pil_crop_square

        if not skip_preload:
            self.preload()

    def __getitem__(self, index: int) -> SampleType:
        if self._first_getitem:
            logger.debug(f"First __getitem__ (idx {index}) - refreshing clients")
            self.refresh_clients()
            self._first_getitem = False

        sample: pd.Series = self.samples.iloc[index]
        image = self._get_image(sample[self.path_key])
        image, crop_coords = self.img_load_fn(image, self.resolution, self.resampling)

        sample = self.apply_data_transforms(sample)

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
        reverse: bool = False,
        shuffle: bool = False,
        no_resize: bool = False,
        fs_type: str | FilesystemType = "s3",
        path_prefix: Optional[str] = None,
        fsspec_kwargs: dict = {},
        pma_schema: Optional[Schema] = None,
        retries: int = 3,
        retry_delay: int = 5,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.prepare_data_per_node = True
        self.mongo_settings = get_mongo_settings(config_path)

        self.dataset = MongoVAEDataset(
            settings=self.mongo_settings,
            resolution=resolution,
            batch_size=batch_size,
            image_key=image_key,
            path_key=path_key,
            extra_keys=extra_keys,
            resampling=resampling,
            reverse=reverse,
            shuffle=shuffle,
            no_resize=no_resize,
            fs_type=fs_type,
            path_prefix=path_prefix,
            fsspec_kwargs=fsspec_kwargs,
            pma_schema=pma_schema,
            retries=retries,
            retry_delay=retry_delay,
            skip_preload=True,
        )

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

    @property
    def batch_size(self):
        return self.dataset.batch_size

    def prepare_data(self) -> None:
        self.dataset.preload()  # runs on local rank 0 to cache metadata

    def setup(self, stage: str):
        self.dataset.preload()  # runs on all ranks
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
