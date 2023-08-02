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
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)
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


class HFDatasetValidation(HFDatasetBase):
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
        super().__init__(dataset, "val", tokenizer, transform, streaming, **kwargs)


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
        self._dataset = dataset
        self._resolution = resolution
        self._tokenizer = tokenizer
        self._streaming = streaming
        self._kwargs = kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not isinstance(self._dataset, HFDataset):
            self._dataset = load_dataset(self._dataset, streaming=self._streaming, **self._kwargs)

        if "train" in self._dataset:
            self._train_dataset = self._dataset["train"]

        if "validation" in self._dataset:
            self._val_dataset = self._dataset["validation"]
        elif "val" in self._dataset:
            self._val_dataset = self._dataset["val"]
        elif "test" in self._dataset:
            self._val_dataset = self._dataset["test"]

    def setup(self, stage: Optional[str] = None):
        self.hf_train = HFDatasetTrain(
            dataset=self._train_dataset,
            resolution=self._resolution,
            tokenizer=self._tokenizer,
            streaming=self._streaming,
        )
        self.hf_val = HFDatasetValidation(
            dataset=self._val_dataset,
            resolution=self._resolution,
            tokenizer=self._tokenizer,
            streaming=self._streaming,
        )

    def train_dataloader(self) -> HFDatasetBase:
        return DataLoader(self.hf_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> HFDatasetBase:
        return DataLoader(self.hf_val, batch_size=self.batch_size, num_workers=self.num_workers)
