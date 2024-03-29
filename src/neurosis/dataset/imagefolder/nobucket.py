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
from neurosis.dataset.utils import clean_word, collate_dict_stack, load_crop_image_file

logger = logging.getLogger(__name__)


class FolderSquareDataset(NoBucketDataset):
    def __init__(
        self,
        *,
        folder: PathLike,
        resolution: int | tuple[int, int] = 256,
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
        super().__init__(resolution)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample: pd.Series = self.samples.iloc[index]
        image, crop_coords = load_crop_image_file(sample.image_path, self.resolution, self.resampling)

        return {
            self.image_key: self.transforms(image),
            self.caption_key: sample.caption,
            "original_size_as_tuple": self._get_osize(sample.resolution),
            "crop_coords_top_left": crop_coords,
            "target_size_as_tuple": self.resolution,
        }

    def preload(self):
        # get paths
        file_iter = self.folder.rglob("**/*.*") if self.recursive else self.folder.glob("*.*")
        # filter to images
        image_files = [x for x in file_iter if x.is_file() and x.suffix.lower() in IMAGE_EXTNS]
        # build dataframe
        self.samples = pd.DataFrame([self.__load_meta(x) for x in image_files]).astype(
            {"image_path": np.bytes_, "caption": np.bytes_, "aspect": np.float32}
        )

    def _get_osize(self, resolution: tuple[int, int]) -> tuple[int, int]:
        return (
            min(resolution[0], self.resolution[0]) if self.clamp_orig else resolution[0],
            min(resolution[1], self.resolution[1]) if self.clamp_orig else resolution[1],
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
        return pd.Series(
            data=[image_path, caption, aspect, resolution],
            index=["image_path", "caption", "aspect", "resolution"],
        )


class FolderSquareModule(LightningDataModule):
    def __init__(
        self,
        folder: PathLike,
        resolution: int | tuple[int, int] = 256,
        batch_size: int = 1,
        image_key: str = "image",
        *,
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

        self.dataset = FolderSquareDataset(
            folder=self.folder,
            recursive=recursive,
            resolution=resolution,
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
