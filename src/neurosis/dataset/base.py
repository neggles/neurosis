import logging
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

logger = logging.getLogger(__name__)


class NoBucketDataset(Dataset):
    def __init__(
        self,
        resolution: int | tuple[int, int],
        batch_size: int = 1,
        *,
        pil_max_image_pixels: Optional[int] = None,
        pil_max_png_bytes: int = 100 * (1024**2),  # 100 MB
    ):
        self.resolution = resolution if isinstance(resolution, tuple) else (resolution, resolution)
        self.batch_size = batch_size

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
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
