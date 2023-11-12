import logging

from torch.utils.data import Sampler

from neurosis.dataset.aspect.base import AspectBucketDataset

logger = logging.getLogger(__name__)


class AspectBucketSampler(Sampler):
    def __init__(self, dataset: AspectBucketDataset, return_bucket: bool = False):
        self.dataset = dataset
        self.return_bucket = return_bucket
        self.batch_iterator = self.dataset.get_batch_iterator(return_bucket=self.return_bucket)

    def __iter__(self):
        return iter(self.batch_iterator)

    def __len__(self):
        return len(self.dataset) // self.dataset.batch_size
