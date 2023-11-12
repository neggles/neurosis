import logging
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import v2 as T

from neurosis.constants import IMAGE_EXTNS
from neurosis.dataset.aspect import AspectBucketList, SDXLBucketList
from neurosis.dataset.aspect.base import AspectBucketDataset
from neurosis.dataset.aspect.bucket import AspectBucket

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
        pil_resample: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
    ):
        super().__init__(buckets, batch_size, image_key, caption_key)
        self.folder = Path(folder).resolve()
        if not (self.folder.exists() and self.folder.is_dir()):
            raise FileNotFoundError(f"Folder {self.folder} does not exist or is not a directory.")

        self.caption_ext = caption_ext
        self.tag_sep = tag_sep
        self.word_sep = word_sep
        self.recursive = recursive
        self.pil_resample = pil_resample
        self.clamp_orig = clamp_orig

        logger.debug(f"Preloading dataset from '{self.folder}' ({recursive=})")
        # load meta
        self._preload()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample: pd.Series = self.samples.iloc[index]
        bucket: AspectBucket = self.buckets[sample.bucket_idx]
        image, crop_coords = self.__load_and_crop(sample.image_path, bucket)

        return {
            self.image_key: image,
            self.caption_key: sample.caption,
            "original_size_as_tuple": (
                min(sample.resolution[0], bucket.width) if self.clamp_orig else sample.resolution[0],
                min(sample.resolution[1], bucket.height) if self.clamp_orig else sample.resolution[1],
            ),
            "crop_coords_top_left": crop_coords,
            "target_size_as_tuple": bucket.size,
        }

    def _preload(self):
        # get paths
        file_iter = self.folder.rglob("**/*.*") if self.recursive else self.folder.glob("*.*")
        # filter to images
        image_files = [x for x in file_iter if x.is_file() and x.suffix.lower() in IMAGE_EXTNS]
        # build dataframe
        self.samples = pd.DataFrame([self.__load_meta(x) for x in image_files]).astype(
            {"image_path": np.string_, "caption": np.string_, "aspect": np.float32, "bucket_idx": np.int32}
        )

        modified = False
        for bucket_id, sample_ids in enumerate(self.bucket2idx.items()):
            n_samples = len(sample_ids)
            if n_samples >= self.batch_size:
                continue
            logger.warn(f"Bucket #{bucket_id} has less than one batch of samples, merging with next bucket.")
            if self.buckets[bucket_id].aspect < 1.0:
                self.samples[sample_ids, "bucket_idx"] = bucket_id + 1

        if modified:
            self._bucket2idx = None
            self._idx2bucket = None

    def __clean_caption(self, caption: str) -> str:
        def clean_word(word: str) -> str:
            if "_" in word:
                word = word.replace("_", self.word_sep)
            if " " in word:
                word = word.replace(" ", self.word_sep)
            return word.strip()

        caption = [clean_word(x) for x in caption.split(", ")]
        return self.tag_sep.join(caption).strip()

    def __load_meta(self, image_path: Path) -> pd.Series:
        caption_file = image_path.with_suffix(self.caption_ext)
        if not caption_file.exists():
            raise FileNotFoundError(f"Caption {self.caption_ext} for image {image_path} does not exist.")
        caption = self.__clean_caption(caption_file.read_text(encoding="utf-8"))
        resolution = np.array(Image.open(image_path).size, np.int32)
        aspect = np.float32(resolution[0] / resolution[1])
        bucket = self.buckets.bucket_idx(aspect)
        return pd.Series(
            data=[image_path, caption, aspect, resolution, bucket],
            index=["image_path", "caption", "aspect", "resolution", "bucket_idx"],
        )

    def __load_and_crop(
        self, image_path: PathLike | bytes, bucket: AspectBucket
    ) -> tuple[Tensor, tuple[int, int]]:
        if isinstance(image_path, bytes):
            image_path = image_path.decode("utf-8")
        # resolve path
        image_path = Path(image_path).resolve()
        # load image
        image = Image.open(image_path)
        # convert to RGB/RGBA if not already (deals with palette images etc.)
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
        # convert RGBA to RGB with white background
        if image.mode == "RGBA":
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        # resize to cover bucket
        image = ImageOps.cover(image, bucket.size, method=self.pil_resample)

        # crop to bucket
        min_edge = min(image.size)
        delta_w, delta_h = image.size[0] - min_edge, image.size[1] - min_edge
        if all([delta_w, delta_h]):
            raise ValueError(f"Failed to crop {image_path} short edge to match {bucket}!")
        top, left = np.random.randint(delta_h + 1), np.random.randint(delta_w + 1)
        image = T.functional.crop(image, top=top, left=left, height=bucket.height, width=bucket.width)
        return image, (top, left)

    def get_batch_iterator(self, return_bucket: bool = False):
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
                    batch.append(indices[k])
                b_offs += 1

            bucket_dict[idx] = (indices, b_len, b_offs)
            yield batch, self.buckets[idx] if return_bucket else batch
