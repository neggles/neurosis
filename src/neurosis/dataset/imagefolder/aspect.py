import logging
from os import PathLike
from pathlib import Path
from typing import Generator, Optional, Sequence

import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from neurosis.constants import IMAGE_EXTNS
from neurosis.dataset.aspect import (
    AspectBucket,
    AspectBucketDataset,
    AspectBucketList,
    AspectDistributedSampler,
    SDXLBucketList,
)
from neurosis.dataset.utils import clean_word, collate_dict_stack, load_bucket_image_file

logger = logging.getLogger(__name__)


class ImageFolderDataset(AspectBucketDataset):
    def __init__(
        self,
        *,
        folder: PathLike,
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
    ):
        super().__init__(buckets)
        self.folder = Path(folder).resolve()
        if not (self.folder.exists() and self.folder.is_dir()):
            raise FileNotFoundError(f"Folder {self.folder} does not exist or is not a directory.")

        self.batch_size = batch_size
        self.image_key = image_key
        self.caption_key = caption_key

        self.caption_ext = caption_ext
        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.recursive = recursive
        self.resampling = resampling
        self.clamp_orig = clamp_orig
        self.process_tags = process_tags
        self.shuffle_tags = shuffle_tags
        self.shuffle_keep = shuffle_keep

        logger.debug(f"Preloading dataset from '{self.folder}' ({recursive=})")
        # load meta
        self.preload()
        self.batch_to_idx: Optional[list[list[int]]] = None
        if batch_size > 1:
            self.batch_to_idx = list(self.get_batch_iterator())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample: pd.Series = self.samples.iloc[index]
        bucket: AspectBucket = self.buckets[sample.bucket_idx]
        image, crop_coords = load_bucket_image_file(sample.image_path, bucket, self.resampling)

        return {
            self.image_key: self.transforms(image),
            self.caption_key: sample.caption,
            "original_size_as_tuple": self._get_osize(sample.resolution, bucket),
            "crop_coords_top_left": crop_coords,
            "target_size_as_tuple": bucket.size,
        }

    def __getitems__(self, indices: Sequence[int] | int) -> dict[str, Tensor | list[Tensor | np.ndarray]]:
        if isinstance(indices, int):
            if self.batch_to_idx is not None:
                indices = self.batch_to_idx[indices]
            else:
                indices = [indices]

        samples = [self.__getitem__(idx) for idx in indices]
        # remap the list of dicts to a dict of lists
        samples = {k: [x[k] for x in samples] for k in samples[0].keys()}
        # collate function can handle stacking tensors from here
        return samples

    def preload(self):
        # get paths
        file_iter = self.folder.rglob("**/*.*") if self.recursive else self.folder.glob("*.*")
        # filter to images
        image_files = [x for x in file_iter if x.is_file() and x.suffix.lower() in IMAGE_EXTNS]
        # build dataframe
        self.samples = pd.DataFrame([self.__load_meta(x) for x in image_files]).astype(
            {"image_path": np.bytes_, "caption": np.bytes_, "aspect": np.float32, "bucket_idx": np.int32}
        )

        modified = False
        for bucket_id, sample_ids in self.bucket2idx.items():
            n_samples = len(sample_ids)
            if n_samples >= self.batch_size:
                continue
            logger.warn(f"Bucket #{bucket_id} has less than one batch of samples, merging with next bucket.")
            if self.buckets[bucket_id].aspect < 1.0:
                self.samples.loc[sample_ids, "bucket_idx"] = bucket_id + 1

        if modified:
            self._bucket2idx = None
            self._idx2bucket = None

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
            data=[str(image_path).encode("utf-8"), caption.encode("utf-8"), aspect, resolution, bucket_idx],
            index=["image_path", "caption", "aspect", "resolution", "bucket_idx"],
        )

    def get_batch_iterator(self) -> Generator[list[int], None, None]:
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


class ImageFolderModule(LightningDataModule):
    def __init__(
        self,
        folder: PathLike,
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
        process_tags: bool = True,
        shuffle_tags: bool = True,
        shuffle_keep: int = 0,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.folder = Path(folder).resolve()

        if not self.folder.exists():
            raise FileNotFoundError(f"Folder {self.folder} does not exist.")
        if not self.folder.is_dir():
            raise ValueError(f"Folder {self.folder} is not a directory.")

        self.dataset = ImageFolderDataset(
            folder=self.folder,
            recursive=recursive,
            buckets=buckets,
            batch_size=batch_size,
            image_key=image_key,
            caption_key=caption_key,
            caption_ext=caption_ext,
            tag_sep=tag_sep,
            word_sep=word_sep,
            resampling=resampling,
            clamp_orig=clamp_orig,
            process_tags=process_tags,
            shuffle_tags=shuffle_tags,
            shuffle_keep=shuffle_keep,
        )

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.sampler: AspectDistributedSampler = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        if self.sampler is None:
            logger.info("Generating sampler")
            self.sampler = AspectDistributedSampler(self.dataset)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=collate_dict_stack,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )
