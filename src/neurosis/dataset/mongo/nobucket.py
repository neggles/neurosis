import logging
from os import PathLike
from typing import Literal, Optional

import numpy as np
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
from neurosis.dataset.utils import clean_word, pil_crop_square

logger = logging.getLogger(__name__)


class MongoSquareDataset(BaseMongoDataset, NoBucketDataset):
    def __init__(
        self,
        settings: MongoSettings,
        *,
        image_key: str = "image",
        caption_key: str = "caption",
        tag_sep: str = ", ",
        word_sep: str = " ",
        process_tags: bool = True,
        shuffle_tags: bool = True,
        shuffle_keep: int = 0,
        clamp_orig: bool = True,
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
        fs_type: str | FilesystemType = "s3",
        path_prefix: Optional[str] = None,
        fsspec_kwargs: dict = {},
        pma_schema: Optional[Schema] = None,
        retries: int = 3,
        retry_delay: int = 5,
        skip_preload: bool = False,
        **kwargs,
    ):
        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.clamp_orig = clamp_orig
        self.process_tags = process_tags
        self.shuffle_tags = shuffle_tags
        self.shuffle_keep = shuffle_keep

        self.image_key = image_key
        self.caption_key = caption_key
        self.batch_keys: list[str] = [
            image_key,
            caption_key,
            "original_size_as_tuple",
            "crop_coords_top_left",
        ]

        BaseMongoDataset.__init__(
            self,
            settings=settings,
            batch_size=batch_size,
            path_key=path_key,
            extra_keys=extra_keys,
            resampling=resampling,
            shuffle=shuffle,
            reverse=reverse,
            data_transforms=data_transforms,
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

        if not skip_preload:
            self.preload()

    def __getitem__(self, index: int) -> SampleType:
        if self._first_getitem:
            self.refresh_clients()
            self._first_getitem = False

        sample: pd.Series = self.samples.iloc[index]
        image = self._get_image(sample[self.path_key])
        image, crop_coords = pil_crop_square(image, self.resolution, self.resampling)

        return {
            self.image_key: self.transforms(image),
            self.caption_key: self.__clean_caption(sample.caption),
            "original_size_as_tuple": self._get_osize((image.width, image.height)),
            "crop_coords_top_left": torch.tensor(crop_coords, dtype=torch.int32),
            **{k: torch.tensor(sample.get(k)) for k in self.extra_keys if k in sample},
        }

    def _get_osize(self, resolution: tuple[int, int]) -> tuple[int, int]:
        if self.clamp_orig:
            resolution = tuple(min(x, y) for x, y in zip(resolution, self.resolution))
        return torch.tensor(resolution, dtype=torch.int32)

    def __clean_caption(self, caption: str | list[str]) -> str:
        if self.process_tags:
            if isinstance(caption, str):
                caption = [clean_word(self.word_sep, x) for x in caption.split(", ")]
            elif isinstance(caption, (list, np.ndarray)):
                caption = [clean_word(self.word_sep, x) for x in caption]
            else:
                raise TypeError(f"Unexpected type for caption: {type(caption)}")

            if self.shuffle_tags:
                if self.shuffle_keep > 0:
                    caption = (
                        caption[: self.shuffle_keep]
                        + np.random.permutation(caption[self.shuffle_keep :]).tolist()
                    )
                else:
                    caption = np.random.permutation(caption).tolist()

            return self.tag_sep.join(caption).strip()
        else:
            if isinstance(caption, list):
                return self.tag_sep.join(caption).strip()
            return caption.strip()


class MongoSquareModule(LightningDataModule):
    def __init__(
        self,
        config_path: PathLike,
        resolution: int | tuple[int, int] = 256,
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        *,
        path_key: str = "s3_path",
        tag_sep: str = ", ",
        word_sep: str = " ",
        extra_keys: list[str] | Literal["all"] = [],
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        reverse: bool = False,
        shuffle: bool = False,
        process_tags: bool = True,
        shuffle_tags: bool = True,
        shuffle_keep: int = 0,
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

        self.dataset = MongoSquareDataset(
            settings=self.mongo_settings,
            resolution=resolution,
            batch_size=batch_size,
            image_key=image_key,
            caption_key=caption_key,
            path_key=path_key,
            tag_sep=tag_sep,
            word_sep=word_sep,
            extra_keys=extra_keys,
            resampling=resampling,
            shuffle=shuffle,
            reverse=reverse,
            process_tags=process_tags,
            shuffle_tags=shuffle_tags,
            shuffle_keep=shuffle_keep,
            fs_type=fs_type,
            path_prefix=path_prefix,
            fsspec_kwargs=fsspec_kwargs,
            pma_schema=pma_schema,
            retries=retries,
            retry_delay=retry_delay,
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
        )
