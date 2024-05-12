import logging
from os import PathLike
from typing import Generator, Literal, Optional

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from PIL import Image
from pymongoarrow.schema import Schema
from torch import Tensor
from torch.utils.data import DataLoader

from neurosis.dataset.aspect import (
    AspectBucket,
    AspectBucketDataset,
    AspectBucketList,
    AspectDistributedSampler,
    SDXLBucketList,
)
from neurosis.dataset.aspect.sampler import AspectBucketSampler
from neurosis.dataset.base import FilesystemType
from neurosis.dataset.mongo.base import BaseMongoDataset
from neurosis.dataset.mongo.settings import MongoSettings, get_mongo_settings
from neurosis.dataset.processing.transform import DataTransform
from neurosis.dataset.utils import (
    clean_word,
    clear_fsspec,
    collate_dict_stack,
    pil_crop_bucket,
    set_s3fs_opts,
)
from neurosis.utils import maybe_collect

logger = logging.getLogger(__name__)


class MongoAspectDataset(BaseMongoDataset, AspectBucketDataset):
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
        # passed to AspectBucketDataset
        buckets: AspectBucketList = SDXLBucketList(),
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
        self.image_key = image_key
        self.caption_key = caption_key
        self.batch_keys: list[str] = [
            image_key,
            caption_key,
            "original_size_as_tuple",
            "crop_coords_top_left",
            "target_size_as_tuple",
        ]

        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.clamp_orig = clamp_orig
        self.process_tags = process_tags
        self.shuffle_tags = shuffle_tags
        self.shuffle_keep = shuffle_keep

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
            fs_type=fs_type,
            path_prefix=path_prefix,
            fsspec_kwargs=fsspec_kwargs,
            pma_schema=pma_schema,
            retries=retries,
            retry_delay=retry_delay,
            **kwargs,
        )
        AspectBucketDataset.__init__(
            self,
            buckets=buckets,
            **kwargs,
        )

        self.batch_to_idx: Optional[list[list[int]]] = None
        if not skip_preload:
            self.preload()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if self._first_getitem:
            self.refresh_clients()
            self._first_getitem = False

        sample: pd.Series = self.samples.iloc[index]
        bucket: AspectBucket = self.buckets[sample.bucket_idx]
        image = self._get_image(sample[self.path_key])
        image, crop_coords = pil_crop_bucket(image, bucket, self.resampling)

        return {
            self.image_key: self.transforms(image),
            self.caption_key: self._clean_caption(sample.caption),
            "original_size_as_tuple": self._get_osize((image.width, image.height), bucket),
            "crop_coords_top_left": torch.tensor(crop_coords, dtype=torch.int32),
            "target_size_as_tuple": torch.tensor(bucket.size, dtype=torch.int32),
            **{k: torch.tensor(sample.get(k)) for k in self.extra_keys if k in sample},
        }

    def preload(self):
        # call the superclasses' preload method
        super().preload()

        # assign aspect ratios to buckets
        self.assign_aspect()

        modified = False
        for bucket_id, sample_ids in self.bucket2idx.items():
            n_samples = len(sample_ids)
            if n_samples >= self.batch_size:
                continue
            logger.warn(f"Bucket #{bucket_id} has less than one batch of samples, merging with next bucket.")
            if self.buckets[bucket_id].aspect < 1.0:
                self.samples.loc[sample_ids, "bucket_idx"] = bucket_id + 1

        if modified is True:
            self._bucket2idx = None
            self._idx2bucket = None

        # make batch_to_idx happen
        if self.batch_size > 1:
            self.batch_to_idx = list(self.get_batch_iterator())

        logger.debug("Preload complete!")
        maybe_collect()

    def assign_aspect(self) -> pd.DataFrame:
        def get_bucket_indices(df: pd.DataFrame):
            aspects: pd.Series = df["aspect"]
            indices: pd.Series = aspects.apply(self.buckets.bucket_idx)
            return indices

        if "bucket_idx" not in self.samples.columns:
            logger.info("Mapping aspect ratios to buckets...")
            self.samples = self.samples.assign(bucket_idx=get_bucket_indices)

    def _get_osize(self, resolution: tuple[int, int], bucket: AspectBucket) -> tuple[int, int]:
        if self.clamp_orig:
            resolution = tuple(min(x, y) for x, y in zip(resolution, bucket.size))
        return torch.tensor(resolution, dtype=torch.int32)

    def _clean_caption(self, caption: str | list[str]) -> str:
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

    def get_batch_iterator(self) -> Generator[list[int], None, None]:
        logger.info("Creating batch iterator")
        max_bucket_len = self.samples.groupby("bucket_idx").size().max()
        index_sched = np.arange(max_bucket_len, dtype=np.int64)
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

        def batch_iterator():
            buckets = bucket_dict.copy()
            for idx in bucket_sched:
                indices, b_len, b_offs = buckets[idx]

                batch = []
                while len(batch) < self.batch_size:
                    k = index_sched[b_offs]
                    if k < b_len:
                        batch.append(indices[k].item())
                    b_offs += 1

                buckets[idx] = (indices, b_len, b_offs)
                yield batch

        return batch_iterator()


class MongoAspectModule(LightningDataModule):
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
        extra_keys: list[str] | Literal["all"] = [],
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        reverse: bool = False,
        shuffle: bool = False,
        clamp_orig: bool = True,
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
        pin_memory: bool = False,
        drop_last: bool = True,
        extra_loader_kwargs: dict = {},
        cache_dir: Optional[PathLike] = None,
    ):
        super().__init__()
        self.prepare_data_per_node = True
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
            extra_keys=extra_keys,
            resampling=resampling,
            shuffle=shuffle,
            reverse=reverse,
            clamp_orig=clamp_orig,
            process_tags=process_tags,
            shuffle_tags=shuffle_tags,
            shuffle_keep=shuffle_keep,
            fs_type=fs_type,
            path_prefix=path_prefix,
            fsspec_kwargs=fsspec_kwargs,
            pma_schema=pma_schema,
            retries=retries,
            retry_delay=retry_delay,
            cache_dir=cache_dir,
            skip_preload=True,
        )

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.extra_loader_kwargs = extra_loader_kwargs

        self.sampler: AspectBucketSampler = None

    @property
    def batch_size(self):
        return self.dataset.batch_size

    def prepare_data(self) -> None:
        self.dataset.preload()  # runs on local rank 0 to cache metadata

    def setup(self, stage: str):
        self.dataset.preload()  # runs on all ranks
        if self.sampler is None:
            logger.info("Generating distributed sampler")
            self.sampler = AspectDistributedSampler(self.dataset)

        logger.debug(f"Refreshing dataset clients for {stage}")
        self.dataset.refresh_clients()

    def train_dataloader(self):
        if self.num_workers > 0:
            return DataLoader(
                self.dataset,
                batch_sampler=self.sampler,
                num_workers=self.num_workers,
                collate_fn=collate_dict_stack,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                worker_init_fn=mongo_worker_init,
                persistent_workers=True,
                **self.extra_loader_kwargs,
            )
        else:
            return DataLoader(
                self.dataset,
                batch_sampler=self.sampler,
                collate_fn=collate_dict_stack,
                pin_memory=self.pin_memory,
                **self.extra_loader_kwargs,
            )


def mongo_worker_init(worker_id: int = -1):
    logger.debug(f"Worker {worker_id} initialiszing")
    clear_fsspec()
    set_s3fs_opts()
