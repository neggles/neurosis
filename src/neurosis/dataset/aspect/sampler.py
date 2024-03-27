import logging
import math
from typing import Iterator, Optional, TypeVar

import torch
from torch.utils.data import DistributedSampler, Sampler

from neurosis.dataset.aspect.base import AspectBucketDataset

T_co = TypeVar("T_co", covariant=True)

logger = logging.getLogger(__name__)


class AspectBucketSampler(Sampler):
    def __init__(self, dataset: AspectBucketDataset):
        self.dataset = dataset
        self.batch_iterator = self.dataset.get_batch_iterator()

    def __iter__(self):
        return iter(self.batch_iterator)

    def __len__(self):
        return len(self.dataset) // self.dataset.batch_size


class AspectDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: AspectBucketDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        batch_indices = list(dataset.get_batch_iterator())

        super().__init__(batch_indices, num_replicas, rank, shuffle, seed, drop_last)

        logger.info(f"rank: {self.rank}, num_samples: {self.num_samples}")

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        if len(indices) != self.total_size:
            raise ValueError(f"Expected indices to have length {self.total_size}, but got {len(indices)}")

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise ValueError(f"Expected indices to have length {self.num_samples}, but got {len(indices)}")

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
