import logging
from io import BytesIO
from os import PathLike
from typing import Callable

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image
from pymongo import MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongoarrow.api import find_pandas_all
from s3fs import S3FileSystem
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader
from torchvision.transforms import v2 as T

from neurosis.dataset.aspect import (
    AspectBucket,
    AspectBucketDataset,
    AspectBucketList,
    AspectBucketSampler,
    SDXLBucketList,
)
from neurosis.dataset.mongo.settings import MongoSettings, get_mongo_settings
from neurosis.dataset.utils import clean_word, pil_crop_bucket, pil_ensure_rgb
from neurosis.utils import maybe_collect

logger = logging.getLogger(__name__)


class MongoAspectDataset(AspectBucketDataset):
    def __init__(
        self,
        settings: MongoSettings,
        buckets: AspectBucketList = SDXLBucketList(),
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        *,
        path_key: str = "s3_path",
        tag_sep: str = ", ",
        word_sep: str = " ",
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
        process_tags: bool = True,
        shuffle_tags: bool = True,
        shuffle_keep: int = 0,
        s3fs_kwargs: dict = {},
    ):
        super().__init__(buckets, batch_size, image_key, caption_key)
        self.settings = settings

        self.path_key = path_key
        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.resampling = resampling
        self.clamp_orig = clamp_orig
        self.process_tags = process_tags
        self.shuffle_tags = shuffle_tags
        self.shuffle_keep = shuffle_keep

        self.client: MongoClient = self.settings.new_client()
        self.fs = S3FileSystem(**s3fs_kwargs)

        # load meta
        logger.debug(
            f"Preloading dataset from mongodb://<host>/{self.settings.database}.{self.settings.collection}"
        )
        self._count: int = None
        self._preload()
        # transforms
        self.transforms: Callable = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
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

    def refresh_client(self):
        """Helper func to replace the current client with a new one."""
        self.client = self.settings.new_client()

    @property
    def collection(self) -> MongoCollection:
        return self.client.get_database(self.settings.db_name).get_collection(self.settings.coll_name)

    def _preload(self):
        if self._count is None:
            logger.info(f"Counting documents in {self.settings.coll_name}")
            self._count = self.settings.count

        if not isinstance(self.samples, pd.DataFrame):
            logger.info(f"Loading metadata for {self._count} documents, this may take a while...")
            self.samples: pd.DataFrame = find_pandas_all(
                self.collection,
                query=dict(self.settings.query.filter),
                **self.settings.query.kwargs,
            )

        if "bucket_idx" not in self.samples.columns:
            logger.info("Mapping aspect ratios to buckets...")
            self.samples = self._assign_aspect(self.samples)

        modified = False
        for bucket_id, sample_ids in enumerate(self.bucket2idx.items()):
            n_samples = len(sample_ids)
            if n_samples >= self.batch_size:
                continue
            logger.warn(f"Bucket #{bucket_id} has less than one batch of samples, merging with next bucket.")
            if self.buckets[bucket_id].aspect < 1.0:
                self.samples[sample_ids, "bucket_idx"] = bucket_id + 1

        if modified is True:
            self._bucket2idx = None
            self._idx2bucket = None

        logger.debug("Preload complete!")
        maybe_collect()

    def _assign_aspect(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_bucket_indices(df: pd.DataFrame):
            aspects: pd.Series = df["aspect"]
            return aspects.apply(self.buckets.bucket_idx)

        return df.assign(bucket_idx=get_bucket_indices)

    def _get_osize(self, resolution: tuple[int, int], bucket: AspectBucket) -> tuple[int, int]:
        if self.clamp_orig:
            return (min(resolution[0], bucket.width), min(resolution[1], bucket.height))
        else:
            return resolution

    def __clean_caption(self, caption: str | list[str]) -> str:
        if self.process_tags:
            if isinstance(caption, list):
                caption = [clean_word(self.word_sep, x) for x in caption]
            else:
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
            if isinstance(caption, list):
                return self.tag_sep.join(caption).strip()
            return caption.strip()

    def _get_image(self, path: str) -> Image.Image:
        image = self.fs.cat(path)
        if not isinstance(image, bytes):
            raise FileNotFoundError(f"Failed to load image from {path}")
        image = Image.open(BytesIO(image))
        image = pil_ensure_rgb(image)

    def get_batch_iterator(self, return_bucket: bool = False):
        logger.info("Creating batch iterator")
        max_bucket_len = self.samples.groupby("bucket_idx").size().max()
        index_sched = np.array(range(max_bucket_len), np.int32)
        np.random.shuffle(index_sched)

        bucket_dict = {
            idx: (frame.index.values, len(frame), 0)
            for idx, frame in self.samples.groupby("bucket_idx")
            if len(frame) >= self.batch_size
        }

        bucket_sched = []
        for idx, (bucket, _, _) in bucket_dict.items():
            bucket_sched.extend([idx] * (len(bucket) // self.batch_size))
        np.random.shuffle(bucket_sched)

        for idx in bucket_sched:
            indices, b_len, b_offs = bucket_dict[idx]

            batch = []
            while len(batch) < self.batch_size:
                k = index_sched[b_offs]
                if k < b_len:
                    batch.append(indices[k].item())
                b_offs += 1

            bucket_dict[idx] = (indices, b_len, b_offs)
            yield (batch, self.buckets[idx]) if return_bucket else batch


class MongoDbModule(LightningDataModule):
    def __init__(
        self,
        config_path: PathLike,
        buckets: AspectBucketList = SDXLBucketList(),
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        *,
        path_key: str = "s3_path",
        tag_sep: str = ", ",
        word_sep: str = " ",
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
        process_tags: bool = True,
        shuffle_tags: bool = True,
        shuffle_keep: int = 0,
        s3fs_kwargs: dict = {},
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.mongo_settings = get_mongo_settings(config_path)

        self.dataset = MongoAspectDataset(
            settings=self.mongo_settings,
            buckets=buckets,
            batch_size=batch_size,
            image_key=image_key,
            caption_key=caption_key,
            path_key=path_key,
            tag_sep=tag_sep,
            word_sep=word_sep,
            resampling=resampling,
            clamp_orig=clamp_orig,
            process_tags=process_tags,
            shuffle_tags=shuffle_tags,
            shuffle_keep=shuffle_keep,
            s3fs_kwargs=s3fs_kwargs,
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.sampler: AspectBucketSampler = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if self.sampler is None:
            logger.info("Generating sampler")
            self.sampler = AspectBucketSampler(self.dataset)

        if stage == "fit":
            logger.info("Refreshing dataset Mongo client")
            self.dataset.refresh_client()

    def train_dataloader(self):
        batch_sampler = BatchSampler(self.sampler, self.dataset.batch_size, self.drop_last)
        return DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )
