from os import PathLike
from pathlib import Path
from typing import Optional

import lightning as L
from gridfs import GridFS
from torch.utils.data import Dataset

from neurosis.dataset.bucket import AspectBucket, AspectBucketList, SDXLBucketList
from neurosis.dataset.mongo.settings import MongoSettings, Query, get_mongo_settings


class MongoAspectDataset(Dataset):
    def __init__(
        self,
        config_path: Optional[PathLike] = None,
        bucket_list: AspectBucketList = SDXLBucketList,
    ):
        if config_path is not None:
            config_path = Path(config_path)

        self.settings: MongoSettings = get_mongo_settings(config_path)
        self.bucket_list = bucket_list

        self.client = self.settings.get_client(new=True)
        self.fs = GridFS(self.settings.get_database(), self.settings.gridfs_prefix)

    def __getitem__(self, item: tuple[int, int, int]):
        pass


class MongoDatasetModule(L.LightningDataModule):
    def __init__(self, config_path, batch_size, resolution, tokenizer, num_workers):
        pass


def get_pixiv_tags():
    pass
