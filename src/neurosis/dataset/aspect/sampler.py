import logging

from torch.utils.data import BatchSampler, Sampler

from neurosis.dataset.aspect.base import AspectBucketDataset

logger = logging.getLogger(__name__)


class AspectBucketSampler(Sampler):
    def __init__(self, dataset: AspectBucketDataset):
        self.dataset = dataset
        self.batch_iterator = self.dataset.get_batch_iterator()

    def __iter__(self):
        return iter(self.batch_iterator)

    def __len__(self):
        return len(self.dataset) // self.dataset.batch_size


class AspectBatchSampler(BatchSampler):
    def __init__(self, dataset: AspectBucketDataset):
        self.dataset = dataset
        self.batch_iterator = self.dataset.get_batch_iterator()

    def __iter__(self):
        return iter(self.batch_iterator)

    def __len__(self):
        return len(self.dataset) // self.dataset.batch_size
