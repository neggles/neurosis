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
        config_path: Optional[PathLike] = None,
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
        process_tags: bool = True,
        shuffle_tags: bool = True,
        shuffle_keep: int = 0,
        path_key: str = "s3_path",
        s3fs_kwargs: dict = {},
    ):
        super().__init__(buckets, batch_size, image_key, caption_key)
        self.settings: MongoSettings = get_mongo_settings(config_path)
        self.client = self.settings.get_client(new=True)

        self.path_key = path_key
        self.caption_ext = caption_ext
        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.recursive = recursive
        self.resampling = resampling
        self.clamp_orig = clamp_orig
        self.process_tags = process_tags
        self.shuffle_tags = shuffle_tags
        self.shuffle_keep = shuffle_keep

        self.fs = S3FileSystem(**s3fs_kwargs)

        # transforms
        self.transforms: Callable = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.samples)

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


class MongoDatasetModule(L.LightningDataModule):
    def __init__(
        self,
        mongo_settings: MongoSettings,
        mongo_query: Query,
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
        pass


def get_pixiv_tags():
    pass
