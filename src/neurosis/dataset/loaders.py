import logging
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd
from PIL import Image
from s3fs import S3FileSystem
from torch import Tensor

from neurosis.constants import MBYTE
from neurosis.dataset.aspect.bucket import AspectBucket
from neurosis.utils import maybe_collect

from .utils import pil_crop_bucket, pil_ensure_rgb

logger = logging.getLogger(__name__)


class S3ImageLoader:
    def __init__(
        self,
        s3fs: Optional[S3FileSystem] = None,
        *,
        s3fs_kwargs: dict = {},
        bucket_name: Optional[str] = None,  # bucket to load images from
        input_key: str = "s3_path",
        output_key: str = "image",
        drop_column: bool = False,  # drop the input key column from the batch after loading
        skip_errors: bool = True,  # skip images that fail to load, by dropping them from the batch
        parallel: bool = False,  # Enable parallelism (requires s3fs_kwargs)
        error_log: Optional[PathLike] = None,  # where to write out images that fail to load (optional)
    ):
        from PIL import Image, PngImagePlugin  # noqa: F401

        if s3fs is None and s3fs_kwargs is None:
            raise ValueError("Either s3fs or s3fs_kwargs must be provided (empty dict is ok)")
        # Assign or create the filesystem object
        if s3fs is None:
            logger.debug("Creating new S3FileSystem object")
            self.fs = S3FileSystem(**s3fs_kwargs)
        elif isinstance(s3fs, S3FileSystem):
            logger.debug("Using provided S3FileSystem object")
            self.fs = s3fs
        else:
            raise TypeError(f"Expected s3fs to be S3FileSystem, got {type(s3fs)}")

        # Save properties for later
        self.bucket_name = bucket_name.rstrip("/") if bucket_name else None
        self.input_key = input_key
        self.output_key = output_key
        self.drop_column = drop_column
        self.skip_errors = skip_errors
        self.parallel = parallel
        self.error_log = Path(error_log) if error_log else None

        # increase PIL limits (only needs to be done once per process)
        Image.init()
        Image.MAX_IMAGE_PIXELS = None
        PngImagePlugin.MAX_TEXT_CHUNK = 8 * MBYTE

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        return self.process_batch(batch)

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        # attach bucket prefix to paths, if necessary
        if self.bucket_name:
            paths = [f"{self.bucket_name}/{x}" for x in batch[self.input_key]]
        else:
            paths = [x for x in batch[self.input_key]]

        # load images from paths
        if self.parallel:
            images = self.get_batched(paths)
        else:
            images = self.get_images(paths)

        # add images to batch (creates a copy)
        batch = batch.assign(**{self.output_key: list(images)})
        # drop the path column if requested
        if self.drop_column:
            batch.drop(columns=[self.input_key], inplace=True)

        # drop any rows with null images
        if self.skip_errors:
            batch.dropna(axis=0, subset=[self.output_key], inplace=True)

        # maybe do a garbage collection
        maybe_collect()
        # return the batch
        return batch

    def get_images(
        self,
        paths: list[str],
        bucket: Optional[AspectBucket] = None,
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
    ) -> Generator[tuple[Tensor, tuple[int, int]] | np.ndarray | None, Any, None]:
        for image_path in paths:
            try:
                image = self._load_image(image_path)
                if bucket:
                    image = pil_crop_bucket(image, bucket, resampling)
                else:
                    image = np.array(image, dtype=np.uint8)
            except Exception as e:
                logger.exception(f"Error loading image from {image_path}")
                if self.error_log:
                    with self.error_log.open("a", encoding="utf-8") as f:
                        f.write(f"{image_path} # {e}\n")
                if self.skip_errors:
                    image = None
                else:
                    raise e
            yield image

    def get_single(self, path: str) -> Image.Image:
        # load from provided FSSpec filesystem
        image_file = self.fs.cat(path, on_error="return")
        if not isinstance(image_file, bytes):
            raise ValueError(f"Failed to load image from {path}")
        image = Image.open(BytesIO(image_file))
        # ensure RGB
        image = pil_ensure_rgb(image)
        # return the image, shockingly enough
        return image

    def get_batched(
        self,
        paths: list[str],
        bucket: Optional[AspectBucket] = None,
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
    ) -> Generator[tuple[Tensor, tuple[int, int]] | np.ndarray | None, Any, None]:
        results: dict[str, bytes] = self.fs.cat(paths, on_error="return")
        for path in paths:
            res = results[path]
            if not isinstance(res, bytes):
                image = (None, (0, 0)) if bucket else None
            else:
                image = Image.open(BytesIO(res))
                image = pil_ensure_rgb(image)
                if bucket:
                    image = pil_crop_bucket(image, bucket, resampling)
                else:
                    image = np.array(image, dtype=np.uint8)
            yield image
