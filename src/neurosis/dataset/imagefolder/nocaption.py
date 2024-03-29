import logging
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from neurosis.constants import IMAGE_EXTNS
from neurosis.dataset.base import NoBucketDataset
from neurosis.dataset.utils import collate_dict_stack, load_crop_image_file

logger = logging.getLogger(__name__)


class FolderVAEDataset(NoBucketDataset):
    def __init__(
        self,
        folder: PathLike,
        resolution: int | tuple[int, int] = 256,
        batch_size: int = 1,
        image_key: str = "image",
        *,
        recursive: bool = False,
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
    ):
        super().__init__(resolution)
        self.folder = Path(folder).resolve()
        if not (self.folder.exists() and self.folder.is_dir()):
            raise FileNotFoundError(f"Folder {self.folder} does not exist or is not a directory.")

        self.batch_size = batch_size
        self.image_key = image_key

        self.recursive = recursive
        self.resampling = resampling
        self.clamp_orig = clamp_orig

        logger.debug(f"Preloading dataset from '{self.folder}' ({recursive=})")
        # load meta
        self.preload()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample: pd.Series = self.samples.iloc[index]
        image, crop_coords = load_crop_image_file(sample.image_path, self.resolution, self.resampling)

        return {
            self.image_key: self.transforms(image),
            "crop_coords_top_left": crop_coords,
        }

    def preload(self):
        # get paths
        file_iter = self.folder.rglob("**/*.*") if self.recursive else self.folder.glob("*.*")
        # filter to images
        image_files = [x for x in file_iter if x.is_file() and x.suffix.lower() in IMAGE_EXTNS]
        # build dataframe
        self.samples = pd.DataFrame([self.__load_meta(x) for x in image_files]).astype(
            {"image_path": np.bytes_, "aspect": np.float32}
        )

    def __load_meta(self, image_path: Path) -> pd.Series:
        resolution = np.array(Image.open(image_path).size, np.int32)
        aspect = np.float32(resolution[0] / resolution[1])
        return pd.Series(
            data=[image_path, aspect, resolution],
            index=["image_path", "aspect", "resolution"],
        )


class FolderVAEModule(LightningDataModule):
    def __init__(
        self,
        folder: PathLike,
        resolution: int | tuple[int, int] = 256,
        batch_size: int = 1,
        image_key: str = "image",
        *,
        recursive: bool = False,
        resampling: Image.Resampling = Image.Resampling.BICUBIC,
        clamp_orig: bool = True,
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

        self.dataset = FolderVAEDataset(
            folder=self.folder,
            recursive=recursive,
            resolution=resolution,
            batch_size=batch_size,
            image_key=image_key,
            resampling=resampling,
            clamp_orig=clamp_orig,
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.dataset.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_dict_stack,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            drop_last=self.drop_last,
        )
