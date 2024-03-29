import logging
from os import PathLike
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader

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


class MemeAspectDataset(AspectBucketDataset):
    def __init__(
        self,
        *,
        num_batches: int,
        folder: PathLike,
        tag_list_path: PathLike = "tag_list.txt",
        buckets: AspectBucketList = SDXLBucketList(),
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        tag_sep: str = ", ",
        word_sep: str = " ",
        recursive: bool = False,
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
        tags_per_img: int = 50,
        prepend_tag: Optional[str] = None,
        prepend_rate: float = 0.0,
    ):
        super().__init__(buckets)
        self.folder = Path(folder).resolve()
        if not (self.folder.exists() and self.folder.is_dir()):
            raise FileNotFoundError(f"Folder {self.folder} does not exist or is not a directory.")

        self.num_samples = num_batches * batch_size
        self.batch_size = batch_size
        self.image_key = image_key
        self.caption_key = caption_key

        tag_list_path = Path(tag_list_path)
        if tag_list_path.is_absolute():
            self.tag_list_path = tag_list_path
        else:
            if tag_list_path.is_file():
                self.tag_list_path = Path(tag_list_path).resolve()
            elif (self.folder / tag_list_path).is_file():
                self.tag_list_path = (self.folder / tag_list_path).resolve()
            else:
                raise FileNotFoundError(f"Tag list file {tag_list_path} does not exist.")

        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.recursive = recursive
        self.resampling = resampling
        self.clamp_orig = clamp_orig

        self.tags_per_img = tags_per_img
        self.prepend_tag = clean_word(self.word_sep, prepend_tag) if prepend_tag is not None else None
        self.prepend_rate = prepend_rate

        logger.debug(f"Preloading dataset from '{self.folder}' ({recursive=})")
        # load meta
        self.preload()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # we straight ignore index and just get the next sample
        if len(self._sample_buffer) == 0:
            logger.debug("Sample buffer is empty, refilling")
            self._sample_buffer = self.samples.sample(frac=1.0).reset_index(drop=True)

        sample: pd.Series = self._sample_buffer[:1].squeeze()
        self._sample_buffer = self._sample_buffer[1:]
        bucket: AspectBucket = self.buckets[sample.bucket_idx]
        image, crop_coords = load_bucket_image_file(sample.image_path, bucket, self.resampling)

        return {
            self.image_key: self.transforms(image),
            self.caption_key: self.__gen_caption(),
            "original_size_as_tuple": self._get_osize(sample.resolution, bucket),
            "crop_coords_top_left": crop_coords,
            "target_size_as_tuple": bucket.size,
        }

    def preload(self):
        # get paths
        file_iter = self.folder.rglob("**/*.*") if self.recursive else self.folder.glob("*.*")
        # filter to images
        image_files = [x for x in file_iter if x.is_file() and x.suffix.lower() in IMAGE_EXTNS]
        # build dataframe
        self.samples = pd.DataFrame([self.__load_meta(x) for x in image_files]).astype(
            {"image_path": np.bytes_, "aspect": np.float32, "bucket_idx": np.int32}
        )
        self._sample_buffer = self.samples.sample(frac=1.0).reset_index(drop=True)

        # aspect bucketize
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

        # load tags
        self.tag_list = [x.strip() for x in self.tag_list_path.read_text().splitlines() if len(x.strip()) > 0]
        self._tag_buffer: list[str] = []

    def _get_osize(self, resolution: tuple[int, int], bucket: AspectBucket) -> tuple[int, int]:
        return (
            min(resolution[0], bucket.width) if self.clamp_orig else resolution[0],
            min(resolution[1], bucket.height) if self.clamp_orig else resolution[1],
        )

    def __gen_caption(self) -> str:
        n = self.tags_per_img

        tags = []
        if len(self._tag_buffer) < n:
            logger.debug(f"Less than {n} tags in buffer, taking the rest and refilling")
            # take all tags from buffer (if any) and decrease n by the amount taken
            tags.extend(self._tag_buffer)
            n = n - len(tags)
            # reshuffle tag list and refill buffer
            self._tag_buffer = np.random.permutation(self.tag_list).tolist()

        # take n tags from the buffer, remove them from the buffer and return
        tags.extend(self._tag_buffer[:n])
        self._tag_buffer = self._tag_buffer[n:]

        # clean tags
        caption = [clean_word(self.word_sep, x) for x in tags]

        # prepend the fixed tag if it exists and the prepend rate is met
        if self.prepend_tag is not None and np.random.rand() < self.prepend_rate:
            if self.prepend_tag in caption:
                caption.remove(self.prepend_tag)
            caption.insert(0, self.prepend_tag)

        return np.bytes_(self.tag_sep.join(caption).strip(), encoding="utf-8")

    def __load_meta(self, image_path: Path) -> pd.Series:
        resolution = np.array(Image.open(image_path).size, np.int32)
        aspect = np.float32(resolution[0] / resolution[1])
        bucket_idx = self.buckets.bucket_idx(aspect)
        return pd.Series(
            data=[image_path, aspect, resolution, bucket_idx],
            index=["image_path", "aspect", "resolution", "bucket_idx"],
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

        for idx in bucket_sched:
            indices, b_len, b_offs = bucket_dict[idx]

            batch = []
            while len(batch) < self.batch_size:
                k = index_sched[b_offs]
                if k < b_len:
                    batch.append(indices[k].item())
                b_offs += 1

            bucket_dict[idx] = (indices, b_len, b_offs)
            yield batch


class MemeAspectModule(LightningDataModule):
    def __init__(
        self,
        folder: PathLike,
        *,
        num_batches: int,
        buckets: AspectBucketList = SDXLBucketList(),
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        tag_list_path: str = "tag_list.txt",
        tag_sep: str = ", ",
        word_sep: str = " ",
        recursive: bool = False,
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
        tags_per_img: int = 50,
        prepend_tag: Optional[str] = None,
        prepend_rate: float = 0.0,
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

        self.dataset = MemeAspectDataset(
            folder=self.folder,
            num_batches=num_batches,
            recursive=recursive,
            buckets=buckets,
            batch_size=batch_size,
            image_key=image_key,
            caption_key=caption_key,
            tag_list_path=tag_list_path,
            tag_sep=tag_sep,
            word_sep=word_sep,
            resampling=resampling,
            clamp_orig=clamp_orig,
            tags_per_img=tags_per_img,
            prepend_tag=prepend_tag,
            prepend_rate=prepend_rate,
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
        batch_sampler = BatchSampler(self.sampler, self.dataset.batch_size, self.drop_last)
        return DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_dict_stack,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
        )
