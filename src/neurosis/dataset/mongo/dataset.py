import logging
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Callable, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image
from pymongo import MongoClient
from s3fs import S3File, S3FileSystem
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from neurosis.dataset.aspect import (
    AspectBucket,
    AspectBucketDataset,
    AspectBucketList,
    AspectBucketSampler,
    SDXLBucketList,
)
from neurosis.dataset.loaders import S3ImageLoader
from neurosis.dataset.mongo.settings import MongoSettings, Query, get_mongo_settings
from neurosis.dataset.utils import clean_word, load_bucket_image_file, pil_crop_bucket, pil_ensure_rgb

logger = logging.getLogger(__name__)


# TODO: add proper support for multiple queries
class MongoAspectDataset(AspectBucketDataset):
    def __init__(
        self,
        settings: MongoSettings,
        buckets: AspectBucketList = SDXLBucketList(),
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        caption_ext: str = ".txt",
        tag_sep: str = ", ",
        word_sep: str = " ",
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
        process_tags: bool = True,
        shuffle_tags: bool = True,
        shuffle_keep: int = 0,
        path_key: str = "s3_path",
        s3fs_kwargs: dict = {},
    ):
        super().__init__(buckets, batch_size, image_key, caption_key)
        self.settings = settings
        self.client: MongoClient = None

        self.path_key = path_key
        self.caption_ext = caption_ext
        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.resampling = resampling
        self.clamp_orig = clamp_orig
        self.process_tags = process_tags
        self.shuffle_tags = shuffle_tags
        self.shuffle_keep = shuffle_keep

        self.fs = S3FileSystem(**s3fs_kwargs)

        # load meta
        logger.debug(
            f"Preloading dataset from mongodb://{self.settings.uri.host}/{self.settings.database}.{self.settings.collection}"
        )
        self._count: int = 0
        self._preload()
        # transforms
        self.transforms: Callable = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return self._count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample: pd.Series = self.samples.iloc[index]
        bucket: AspectBucket = self.buckets[sample.bucket_idx]
        image = self._get_image(sample[self.path_key])
        image, crop_coords = pil_crop_bucket(image, bucket, self.resampling)

        return {
            self.image_key: self.transforms(image),
            self.caption_key: self.__clean_caption(sample.caption),
            "original_size_as_tuple": self._get_osize(sample.resolution, bucket),
            "crop_coords_top_left": crop_coords,
            "target_size_as_tuple": bucket.size,
        }

    def _preload(self):
        logger.debug(f"Counting documents... filter: {self.settings.query.filter}")
        self._count = self.settings.get_count()

    def _get_image(self, path: str) -> Image.Image:
        image = self.fs.cat(path)
        if not isinstance(image, bytes):
            raise FileNotFoundError(f"Failed to load image from {path}")
        image = Image.open(BytesIO(image))
        image = pil_ensure_rgb(image)

    def _get_osize(self, resolution: tuple[int, int], bucket: AspectBucket) -> tuple[int, int]:
        return (
            min(resolution[0], bucket.width) if self.clamp_orig else resolution[0],
            min(resolution[1], bucket.height) if self.clamp_orig else resolution[1],
        )

    def __clean_caption(self, caption: str) -> str:
        if self.process_tags:
            caption = [clean_word(self.word_sep, x) for x in caption.split(", ")]

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
            return caption.strip()

    def __load_meta(self, image_path: Path) -> pd.Series:
        caption_file = image_path.with_suffix(self.caption_ext)
        if not caption_file.exists():
            raise FileNotFoundError(f"Caption {self.caption_ext} for image {image_path} does not exist.")

        caption = self.__clean_caption(caption_file.read_text(encoding="utf-8"))
        resolution = np.array(Image.open(image_path).size, np.int32)
        aspect = np.float32(resolution[0] / resolution[1])
        bucket_idx = self.buckets.bucket_idx(aspect)
        return pd.Series(
            data=[image_path, caption, aspect, resolution, bucket_idx],
            index=["image_path", "caption", "aspect", "resolution", "bucket_idx"],
        )


class MongoDatasetModule(LightningDataModule):
    def __init__(
        self,
        config_path: Optional[PathLike] = None,
        *,
        buckets: AspectBucketList = SDXLBucketList(),
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        caption_ext: str = ".txt",
        tag_sep: str = ", ",
        word_sep: str = " ",
        recursive: bool = False,
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
        num_workers: int = 0,
    ):
        super().__init__()
        self.mongo_settings = get_mongo_settings(config_path)

        self.dataset = MongoAspectDataset(
            self.mongo_settings,
            buckets,
            batch_size,
            image_key,
            caption_key,
            caption_ext,
            tag_sep,
            word_sep,
            recursive,
            resampling,
            clamp_orig,
        )
        self.sampler = AspectBucketSampler(self.dataset)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def get_pixiv_tags():
    pass
