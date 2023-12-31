import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from neurosis.dataset.aspect.lists import AspectBucketList

logger = logging.getLogger(__name__)


class AspectBucketDataset(Dataset):
    def __init__(
        self,
        buckets: AspectBucketList,
        batch_size: int = 1,
        image_key: str = "image",
        caption_key: str = "caption",
        *,
        pil_max_image_pixels: Optional[int] = None,
        pil_max_png_bytes: int = 100 * (1024**2),  # 100 MB
    ):
        self.buckets = buckets
        self.batch_size = batch_size

        # be quiet, PIL
        Image.MAX_IMAGE_PIXELS = pil_max_image_pixels
        PngImagePlugin.MAX_TEXT_CHUNK = pil_max_png_bytes

        # Assign output keys if not already set
        if not hasattr(self, "image_key"):
            self.image_key = image_key
        if not hasattr(self, "caption_key"):
            self.caption_key = caption_key

        # Set up the DataFrame
        self.samples: pd.DataFrame = None
        self._bucket2idx: dict[int, np.ndarray] = None
        self._idx2bucket: dict[int, int] = None

        # set default transforms
        self.transforms: Callable = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @property
    def bucket2idx(self) -> dict[int, np.ndarray]:
        if self.samples is None:
            raise ValueError("Cannot access bucket2idx before dataset is loaded.")
        if self._bucket2idx is None:
            self._bucket2idx = {idx: frame.index.values for idx, frame in self.samples.groupby("bucket_idx")}
        return self._bucket2idx

    @property
    def idx2bucket(self) -> dict[int, int]:
        if self.samples is None:
            raise ValueError("Cannot access idx2bucket before dataset is loaded.")
        if self._idx2bucket is None:
            self._idx2bucket = {idx: bucket for bucket, idx in self.bucket2idx.items()}
        return self._idx2bucket

    def get_batch_iterator(self):
        ...
