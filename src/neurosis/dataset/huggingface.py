import logging
from os import PathLike
from typing import Callable, Optional, Tuple, Union

import lightning as L
from datasets import (
    Dataset as HFDataset,
    load_dataset,
)
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class HFDatasetBase(Dataset):
    def __init__(
        self,
        dataset: Union[str, HFDataset],
        split: str = "train",
        tokenizer: Optional[Union[str, PathLike, PreTrainedTokenizer]] = None,
        transform: Optional[Callable] = None,
        image_key: str = "image",
        caption_key: str = "caption",
        streaming: bool = False,
        tokenizer_kwargs: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
        # set streaming
        self.image_key: str = image_key
        self.caption_key: str = caption_key
        self._streaming = streaming

        # load dataset
        if isinstance(dataset, HFDataset):
            self.dataset: HFDataset = dataset
        else:
            self.dataset: HFDataset = load_dataset(dataset, split=split, streaming=streaming, **kwargs)

        # load tokenizer if provided
        if isinstance(tokenizer, PreTrainedTokenizer):
            self.tokenizer: PreTrainedTokenizer = tokenizer
        elif isinstance(tokenizer, (str, PathLike)):
            try:
                self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    tokenizer, **tokenizer_kwargs
                )
            except OSError:
                self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    tokenizer, subfolder="tokenizer", **tokenizer_kwargs
                )
        else:
            self.tokenizer = None

        # assign transforms callable
        self.transform: T.Compose = transform

        # set length from meta if streaming
        if self._streaming:
            self._length: int = self.dataset.info.splits[split].num_examples

    def __len__(self) -> int:
        if self._streaming:
            return self._length
        if hasattr(self.dataset, "num_rows"):
            return self.dataset.num_rows
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self.dataset[idx]

        image: Image.Image = sample[self.image_key]
        if self.transform is not None:
            image = self.transform(image)

        caption = sample[self.caption_key]
        if isinstance(caption, list):
            caption: str = " ".join(caption)

        if self.tokenizer is not None:
            caption = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            )
            return {"image": image, "caption": caption}
        else:
            return {"image": image}


class HFDatasetTrain(HFDatasetBase):
    def __init__(
        self,
        dataset: Union[str, HFDataset],
        resolution: Union[Tuple[int, int], int] = 256,
        tokenizer: Optional[Union[str, PathLike, PreTrainedTokenizer]] = None,
        streaming: bool = False,
        **kwargs,
    ) -> None:
        transform = T.Compose(
            [
                T.Resize(resolution),
                T.RandomCrop(resolution),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        super().__init__(dataset, "train", tokenizer, transform, streaming, **kwargs)


class HfDatasetEvaluation(HFDatasetBase):
    def __init__(
        self,
        dataset: Union[str, HFDataset],
        resolution: Union[Tuple[int, int], int] = 256,
        tokenizer: Optional[Union[str, PathLike, PreTrainedTokenizer]] = None,
        streaming: bool = False,
        **kwargs,
    ) -> None:
        transform = T.Compose(
            [
                T.Resize(resolution),
                T.CenterCrop(resolution),
                T.ToTensor(),
            ]
        )
        super().__init__(dataset, "eval", tokenizer, transform, streaming, **kwargs)


class HFDatasetModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Union[str, HFDataset],
        batch_size: int = 1,
        resolution: Union[Tuple[int, int], int] = 256,
        tokenizer: Optional[Union[str, PathLike, PreTrainedTokenizer]] = None,
        streaming: bool = False,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self._dataset = dataset if isinstance(dataset, HFDataset) else str(dataset).lstrip("\\~/.")
        self._resolution = resolution
        self._tokenizer = tokenizer
        self._streaming = streaming
        self._kwargs = kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_dataset = None
        self._hf_train = None
        self._test_dataset = None
        self._hf_test = None
        self._predict_dataset = None
        self._hf_predict = None

    def prepare_data(self):
        if not isinstance(self._dataset, HFDataset):
            self._dataset = load_dataset(self._dataset, streaming=self._streaming, **self._kwargs)

        for key in self._dataset.keys():
            match key:
                case val if key in ["train", "training", "fit"]:
                    self._train_dataset = self._dataset[val]
                case val if key in ["val", "validate", "eval", "evaluation"]:
                    self._test_dataset = self._test_dataset or self._dataset[val]
                case val if key in ["test", "testing"]:
                    self._test_dataset = self._dataset[val]
                case _:
                    logger.exception(f"Unknown dataset split {key} will be ignored!")

        if self._test_dataset is None:
            logger.warning("No test dataset found, using training dataset instead")
            self._test_dataset = self._train_dataset
        if self._val_dataset is None:
            self._val_dataset = self._train_dataset
        if self._predict_dataset is None:
            self._predict_dataset = self._train_dataset

    def setup(self, stage: Optional[str] = None):
        self._hf_train = HFDatasetTrain(
            dataset=self._train_dataset,
            resolution=self._resolution,
            tokenizer=self._tokenizer,
            streaming=self._streaming,
        )
        self._hf_test = HfDatasetEvaluation(
            dataset=self._test_dataset,
            resolution=self._resolution,
            tokenizer=self._tokenizer,
            streaming=self._streaming,
        )
        self._hf_predict = HfDatasetEvaluation(
            dataset=self._predict_dataset,
            resolution=self._resolution,
            tokenizer=self._tokenizer,
            streaming=self._streaming,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._hf_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._hf_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._hf_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        """Returns a dataloader for prediction. This is just the training dataloader but only text embeds."""
        return DataLoader(self._hf_train, batch_size=self.batch_size, num_workers=self.num_workers)
