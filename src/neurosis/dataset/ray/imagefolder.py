import io
import logging
import warnings
from dataclasses import dataclass
from itertools import chain
from math import ceil
from os import PathLike, getenv
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ray
from fsspec import AbstractFileSystem
from gcsfs import GCSFileSystem
from humanfriendly import format_size, format_timespan
from ray import data
from ray._private.worker import RayContext
from ray.actor import ActorHandle
from ray.data import read_datasource
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.util import _check_import
from ray.data.context import DataContext
from ray.data.dataset import Dataset
from ray.data.datasource import (
    BaseFileMetadataProvider,
    BinaryDatasource,
    FastFileMetadataProvider,
    FileExtensionFilter,
    Partitioning,
    PathPartitionFilter,
    Reader,
)
from ray.data.datasource.image_datasource import _ImageDatasourceReader, _ImageFileMetadataProvider
from ray.runtime_env import RuntimeEnv

if TYPE_CHECKING:
    import pyarrow
    from PIL import Image
    from ray.data.block import T


class TaggedImageDatasource(BinaryDatasource):
    """A datasource that lets you read images with matching tag data files next to them."""

    _FILE_EXTENSION = ["png", "jpg", "jpeg", "webp", "avif"]

    def create_reader(
        self,
        mode: Optional[str] = None,
        caption_ext: str = ".txt",
        max_edge: Optional[int] = None,
        include_paths: bool = True,
        include_post_id: bool = False,
        **reader_args,
    ) -> "Reader[T]":
        self.include_paths = include_paths
        self.caption_ext = caption_ext

        if max_edge is not None and max_edge < 0:
            raise ValueError(f"Expected `max_edge` to be a positive integer, but got {max_edge} instead.")
        self.size = (max_edge, max_edge) if max_edge is not None else None

        self.mode = mode or "RGB"
        if self.mode not in ["RGB", "RGBA", "BGR;24"]:
            raise ValueError(f"Got unsupported `mode` {mode}, must be 'RGB', 'RGBA', or 'BGR;24'")

        _check_import(self, module="PIL", package="pillow")

        return _ImageDatasourceReader(
            self,
            size=self.size,
            mode="RGB" if self.mode == "BGR;24" else self.mode,
            include_paths=True,
            include_post_id=include_post_id,
            **reader_args,
        )

    def _read_file(
        self,
        f: "pyarrow.NativeFile",
        path: str,
        include_paths: bool,
        include_post_id: bool,
        **reader_args,
    ) -> "pyarrow.Table":
        from PIL import Image  # noqa: F811

        Image.MAX_IMAGE_PIXELS = None

        # get image
        records = super()._read_file(f, path, include_paths=True, **reader_args)
        assert len(records) == 1
        path, image_data = records[0]

        # get caption file
        caption_path = Path(path).with_suffix(self.caption_ext)

        # decode image with Pillow
        image = Image.open(io.BytesIO(image_data))

        # first make everything be RGB or RGBA
        if image.mode not in ["RGB", "RGBA", "BGR;24", self.mode]:
            if "transparency" in image.info:
                image = image.convert("RGBA")
            else:
                image = image.convert("RGB")

        # delete transparency unless explicitly requested
        if image.mode == "RGBA" and self.mode != "RGBA":
            # convert RGBA to RGB with white background
            canvas = Image.new("RGBA", image.size, (255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")

        # resize
        if self.size is not None:
            image = self._smart_resize(image)
        # convert
        if image.mode != self.mode:
            image = image.convert(self.mode)
        # be numpy
        image = np.array(image)

        # send block back to parent class
        builder = DelegatingBlockBuilder()
        item = {"image": image}
        # add path and/or post_id if requested
        if include_paths:
            item.update({"path": path})
        if include_post_id:
            item.update({"post_id": Path(path).stem})

        builder.add(item)
        block = builder.build()

        return block

    @classmethod
    def file_extension_filter(cls) -> PathPartitionFilter:
        return FileExtensionFilter(cls._FILE_EXTENSION)
