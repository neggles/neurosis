import gc
import logging
from io import BytesIO
from os import PathLike
from typing import Any, Optional

import torch
from adlfs import AzureBlobFileSystem
from azure.identity.aio import DefaultAzureCredential
from lightning.pytorch.plugins.io import TorchCheckpointIO
from torch.serialization import MAP_LOCATION
from typing_extensions import override

logger = logging.getLogger(__name__)


class BlobCheckpointIO(TorchCheckpointIO):
    def __init__(self, account_name: str, anon: bool = False) -> None:
        super().__init__()
        self.account_name = account_name
        self.anon = anon
        self.cred = DefaultAzureCredential()
        self._fs: Optional[AzureBlobFileSystem] = None

    @property
    def fs(self) -> AzureBlobFileSystem:
        if self._fs is None:
            self._fs = AzureBlobFileSystem(
                account_name=self.account_name,
                credential=self.cred,
                anon=self.anon,
            )
        return self._fs

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: PathLike, storage_options: Optional[Any] = None
    ) -> None:
        buf = BytesIO()
        logger.debug("Serializing checkpoint")
        torch.save(checkpoint, buf)
        with self.fs.open(path, "wb") as f:
            logger.debug("Writing checkpoint to blob storage: {path}")
            f.write(buf.getvalue())
        del buf
        gc.collect()

    @override
    def load_checkpoint(self, path: PathLike, map_location: Optional[MAP_LOCATION] = None) -> dict[str, Any]:
        if not self.fs.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        with self.fs.open(path, "rb") as f:
            logger.debug(f"Loading checkpoint from blob storage: {path}")
            buf = BytesIO(f.read())
        logger.debug("Deserializing checkpoint")
        return torch.load(buf, map_location=map_location)

    @override
    def remove_checkpoint(self, path: PathLike) -> None:
        if self.fs.isfile(path):
            self.fs.rm(path, recursive=True)
            logger.debug(f"Removed checkpoint: {path}")
