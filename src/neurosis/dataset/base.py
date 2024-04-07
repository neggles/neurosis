import logging
from enum import Enum
from typing import Callable, Optional, TypeAlias

import numpy as np
import pandas as pd
import torch
from PIL import Image, PngImagePlugin
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

logger = logging.getLogger(__name__)

SampleType: TypeAlias = dict[str, Tensor | np.ndarray] | pd.Series
BatchType: TypeAlias = dict[str, list[Tensor | np.ndarray | pd.Series]] | pd.DataFrame


class FilesystemType(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


class NoBucketDataset(Dataset):
    def __init__(
        self,
        resolution: int | tuple[int, int],
        pil_max_image_pixels: Optional[int] = None,
        pil_max_png_bytes: int = 100 * (1024**2),  # 100 MB
        **kwargs,
    ):
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)

        # be quiet, PIL
        Image.MAX_IMAGE_PIXELS = pil_max_image_pixels
        PngImagePlugin.MAX_TEXT_CHUNK = pil_max_png_bytes

        # Set up the DataFrame
        self.samples: pd.DataFrame = None

        # set default transforms
        self.transforms: Callable = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.5], [0.5]),
            ]
        )
