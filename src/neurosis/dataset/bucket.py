from collections import UserList
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
from torchvision import transforms as T


@dataclass
class AspectBucket:
    width: int
    height: int
    indices: list[int] = field(default_factory=list)

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
        return f"AspectBucket({self.width}, {self.height}) ({self.aspect:.2f}:1, {self.pixels // 1000}K px)"

    def __hash__(self) -> int:
        return hash((self.width, self.height))

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
        min_pixels: Optional[int] = None,
        oversize_pct: float = 2.5,
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

        self.n_buckets = n_buckets
        self.edge_min = edge_min
        self.edge_max = edge_max
        self.edge_step = edge_step
        self.max_aspect = max_aspect if max_aspect > 0.0 else float("inf")
        self.max_pixels = int(tgt_pixels * (1.0 + (oversize_pct / 100)))
        self.min_pixels = min_pixels if min_pixels is not None else (edge_min + edge_step) ** 2

        self.data: list[AspectBucket] = []
        self._generate()

    def _generate(self) -> None:
        # Get all valid edge lengths
        valid_edge_px = [x for x in range(self.edge_min, self.edge_max + 1, self.edge_step)]

        # Generate all potentially-valid buckets
        valid_buckets = [
            AspectBucket(x, y)
            for x, y in product(valid_edge_px, valid_edge_px)
            if all((x >= y, self.min_pixels <= (x * y) <= self.max_pixels, x / y <= self.max_aspect))
        ]

        # Group buckets by aspect ratio (rounding to 2 decimal places)
        buckets_by_aspect = {}
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

    def bucket(self, ratio: float) -> AspectBucket:
        """Returns the bucket with the closest aspect ratio to the given ratio"""
        if ratio < 0.0 or ratio > self.max_aspect:
            raise ValueError(f"ratio must be in [0.0, {self.max_aspect}], got {ratio}")
        return min(self.data, key=lambda x: abs(x.aspect - ratio))

    @property
    def ratios(self) -> list[float]:
        return [bucket.aspect for bucket in self.data]


class SDXLBucketList(AspectBucketList):
    """Hard-coded bucket list matching original SDXL training configuration"""

    def __init__(self):
        self.data: list[AspectBucket] = [
            AspectBucket(512, 2048),
            AspectBucket(512, 1984),
            AspectBucket(512, 1920),
            AspectBucket(512, 1856),
            AspectBucket(576, 1792),
            AspectBucket(576, 1728),
            AspectBucket(576, 1664),
            AspectBucket(640, 1600),
            AspectBucket(640, 1536),
            AspectBucket(704, 1472),
            AspectBucket(704, 1408),
            AspectBucket(704, 1344),
            AspectBucket(768, 1344),
            AspectBucket(768, 1280),
            AspectBucket(832, 1216),
            AspectBucket(832, 1152),
            AspectBucket(896, 1152),
            AspectBucket(896, 1088),
            AspectBucket(960, 1088),
            AspectBucket(960, 1024),
            # 2nd half of list
            AspectBucket(1024, 1024),
            AspectBucket(1024, 960),
            AspectBucket(1088, 960),
            AspectBucket(1088, 896),
            AspectBucket(1152, 896),
            AspectBucket(1152, 832),
            AspectBucket(1216, 832),
            AspectBucket(1280, 768),
            AspectBucket(1344, 768),
            AspectBucket(1408, 704),
            AspectBucket(1472, 704),
            AspectBucket(1536, 640),
            AspectBucket(1600, 640),
            AspectBucket(1664, 576),
            AspectBucket(1728, 576),
            AspectBucket(1792, 576),
            AspectBucket(1856, 512),
            AspectBucket(1920, 512),
            AspectBucket(1984, 512),
            AspectBucket(2048, 512),
        ]

        self.n_buckets = len(self.data)
        self.edge_min = 512
        self.edge_max = 2048
        self.edge_step = 64
        self.max_aspect = 4.0
        self.max_pixels = max((bucket.pixels for bucket in self.data))
        self.min_pixels = min((bucket.pixels for bucket in self.data))
