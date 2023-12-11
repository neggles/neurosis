import logging
from bisect import bisect_left
from collections import UserList
from dataclasses import dataclass, field
from itertools import product
from math import sqrt
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def percent_diff(v1: int, v2: int) -> float:
    return round((v1 - v2) / ((v1 + v2) / 2) * 100, 2)


@dataclass
class AspectBucket:
    width: int
    height: int
    square_px: Optional[int] = field(default=None, repr=False)
    error: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.width % 32 != 0 or self.height % 32 != 0:
            raise ValueError(f"width and height must be multiples of 32, got {self.width} and {self.height}")
        if self.square_px:
            self.error = percent_diff(self.width * self.height, self.square_px**2)

    @property
    def aspect(self) -> float:
        return round(self.width / self.height, 4)

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns (height, width, 3) (numpy convention, but not an ndarray)"""
        return (self.height, self.width, 3)

    @property
    def size(self) -> tuple[int, int]:
        """Returns (width, height) (PIL convention)"""
        return (self.width, self.height)

    def __repr__(self) -> str:
        return (
            f"AspectBucket({self.width}, {self.height}) "
            + f"({self.aspect:.2f}:1, {self.pixels // 1000}K px"
            + f"{f', {self.error:.2f}% error' if self.error else ''}"
            + ")"
        )

    def __hash__(self) -> int:
        return hash((self.width, self.height, self.square_px or 0))

    @classmethod
    def flipped(cls, bucket: "AspectBucket") -> "AspectBucket":
        return cls(bucket.height, bucket.width)

    @classmethod
    def select_by_px(cls, buckets: list["AspectBucket"], alt: bool = False) -> "AspectBucket":
        """Selects the bucket with the most pixels, or the second-most if alt=True"""
        if len(buckets) > 1:
            buckets = sorted(buckets, key=lambda x: x.pixels)
            return buckets[-2] if alt else buckets[-1]
        elif len(buckets) == 1:
            return buckets[0]  # Only one bucket, return it
        else:
            raise ValueError("Cannot select from empty list of buckets")

    def resize(self, image: Image.Image, method: Image.Resampling = Image.Resampling.BICUBIC):
        return ImageOps.cover(image, self.size, method=method)


class AspectBucketList(UserList):
    """Aspect bucket list generated from a set of constraints"""

    def __init__(
        self,
        n_buckets: int = 25,
        edge_min: int = 512,
        edge_max: int = 1536,
        edge_step: int = 64,
        max_aspect: float = 2.5,
        tgt_pixels: int = 1024 * 1024,
        tolerance: float = 5,
        bias_square: bool = True,
        use_atan: bool = False,
    ):
        if n_buckets < 1 or n_buckets > 100:
            raise ValueError(f"n_buckets must be in [1, 100], got {n_buckets}")
        if edge_min < edge_step or edge_min > edge_max:
            raise ValueError(f"edge_min must be in [edge_step, edge_max], got {edge_min}")
        if edge_max < edge_min or edge_max > 4096:
            raise ValueError(f"edge_max must be in [edge_min, 4096], got {edge_max}")
        if edge_max % edge_step != 0 or edge_min % edge_step != 0:
            raise ValueError(f"min and max must be multiples of step, got {edge_min} and {edge_max}")
        if edge_max // edge_min < max_aspect:
            raise ValueError(f"max_aspect must be less than edge_max / edge_min, got {max_aspect}")

        self.data: list[AspectBucket]
        self.n_buckets = n_buckets
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.edge_step = edge_step
        self.max_aspect = max_aspect if max_aspect > 0.0 else float("inf")
        self.max_pixels = int(tgt_pixels * (1.0 + (tolerance / 100)))
        self.min_pixels = int(tgt_pixels * (1.0 - (tolerance / 100)))
        self.bias_square = bias_square
        self.use_atan = use_atan
        self._square_px = int(sqrt(tgt_pixels)) if sqrt(tgt_pixels).is_integer() else None

        # don't generate buckets if we're a predefined list subclass
        if not hasattr(self, "data"):
            self._generate()

    def _generate(self) -> None:
        # make sure this isn't a reroll
        if hasattr(self, "data") and self.data is not None:
            raise ValueError("Buckets were predefined or have already been generated!")

        # Get all valid edge lengths
        valid_edge_px = [x for x in range(self.edge_min, self.edge_max + 1, self.edge_step)]

        # Generate all potentially-valid buckets
        valid_buckets = [
            AspectBucket(x, y, square_px=self._square_px)
            for x, y in product(valid_edge_px, valid_edge_px)
            if all((x >= y, self.min_pixels <= (x * y) <= self.max_pixels, x / y <= self.max_aspect))
        ]

        # Group buckets by aspect ratio (rounding to 2 decimal places)
        buckets_by_aspect: dict[float, list[AspectBucket]] = {}
        for bucket in valid_buckets:
            aspect = round(bucket.aspect, 2)
            if aspect not in buckets_by_aspect:
                buckets_by_aspect[aspect] = []
            buckets_by_aspect[aspect].append(bucket)

        # Sort each group by pixel count, select the largest bucket from each group,
        # then sort the groups by aspect ratio. Square (train dim) bucket will be the first.
        unique_buckets = sorted(
            [AspectBucket.select_by_px(buckets) for buckets in buckets_by_aspect.values()],
            key=lambda x: x.aspect,
        )
        # Make sure we have enough buckets, if not lets go back for more
        if len(unique_buckets) < self.n_buckets:
            more_buckets = sorted(
                [AspectBucket.select_by_px(buckets, alt=True) for buckets in buckets_by_aspect.values()],
                key=lambda x: x.aspect,
            )
            unique_buckets.extend(more_buckets)
            # if we still don't have enough buckets, give up
            if len(unique_buckets) < self.n_buckets:
                raise ValueError(
                    f"{self.n_buckets} buckets requested but only {len(unique_buckets)} buckets generated. "
                    + "Try reducing edge_step or edge_min, or increasing edge_max."
                )

        # We want a distribution of buckets that is roughly linear in log space.
        # We do this by taking the log of the pixel count and then dividing by the number of buckets.
        bucket_split = np.clip((self.n_buckets + 1) // 2, 1, len(unique_buckets))
        bucket_indices = np.linspace(0, len(unique_buckets) - 1, bucket_split, dtype=int).tolist()

        # Grab the buckets at the indices we want, and their flipped versions, ensuring that
        # we don't have any duplicates. Sort by aspect ratio.
        buckets = sorted(
            {
                *(unique_buckets[i] for i in bucket_indices),
                *(AspectBucket.flipped(unique_buckets[i]) for i in bucket_indices),
            },
            key=lambda x: x.aspect,
        )

        # Save buckets for later.
        self.data = buckets

    def __getitem__(self, i: int | slice) -> Union[AspectBucket, "AspectBucketList"]:
        return self.data[i]

    def bucket_idx(self, ratio: float) -> int:
        """Returns the index of the bucket with the closest aspect ratio to the given ratio"""
        if ratio < 0.0:
            raise ValueError(f"ratio must be > 0, got {ratio}")
        return self.__bucket(ratio, return_index=True)

    def bucket(self, ratio: float) -> AspectBucket:
        """Returns the bucket with the closest aspect ratio to the given ratio"""
        if ratio < 0.0:
            raise ValueError(f"ratio must be > 0, got {ratio}")
        return self.__bucket(ratio, return_index=False)

    def __bucket(self, ratio: float, return_index: bool = False):
        """Get the actual bucket, or just the index of it."""
        # if square just return the square bucket
        if ratio == 1.0:
            bucket_idx = self.ratios.index(1.0)
            return bucket_idx if return_index else self.data[bucket_idx]

        # optionally use arctan rather than raw aspect ratio
        find_ratio = np.arctan(ratio) if self.use_atan else ratio
        aspect_list = self.arctans if self.use_atan else self.ratios

        if self.bias_square:
            # Choose closest bucket, biasing towards aspect 1.0 (square) so the bucket will
            # always fit within the rescaled image dimensions
            bucket_idx = bisect_left(aspect_list, find_ratio)
            if ratio > 1.0:
                bucket_idx -= 1
        else:
            # This avoids the incorrect aspect ratio bias used by the original NAI implementation.
            bucket_idx = np.interp(find_ratio, aspect_list, self.indices).round().astype(int)

        return bucket_idx if return_index else self.data[bucket_idx]

    @property
    def ratios(self) -> list[float]:
        return [bucket.aspect for bucket in self.data]

    @property
    def arctans(self) -> list[float]:
        return [np.arctan(bucket.aspect) for bucket in self.data]

    @property
    def indices(self) -> list[int]:
        return list(range(len(self.ratios)))


class AspectBucketMapper:
    def __init__(
        self,
        buckets: AspectBucketList,
    ):
        self.buckets = buckets
