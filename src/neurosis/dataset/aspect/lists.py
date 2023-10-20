from neurosis.dataset.aspect.bucket import AspectBucket, AspectBucketList


class SDXLBucketList(AspectBucketList):
    """Hard-coded bucket list matching original SDXL training configuration"""

    _TRAIN_RES = 1024

    def __init__(self):
        self.data: list[AspectBucket] = [
            AspectBucket(512, 2048, self._TRAIN_RES),
            AspectBucket(512, 1984, self._TRAIN_RES),
            AspectBucket(512, 1920, self._TRAIN_RES),
            AspectBucket(512, 1856, self._TRAIN_RES),
            AspectBucket(576, 1792, self._TRAIN_RES),
            AspectBucket(576, 1728, self._TRAIN_RES),
            AspectBucket(576, 1664, self._TRAIN_RES),
            AspectBucket(640, 1600, self._TRAIN_RES),
            AspectBucket(640, 1536, self._TRAIN_RES),
            AspectBucket(704, 1472, self._TRAIN_RES),
            AspectBucket(704, 1408, self._TRAIN_RES),
            AspectBucket(704, 1344, self._TRAIN_RES),
            AspectBucket(768, 1344, self._TRAIN_RES),
            AspectBucket(768, 1280, self._TRAIN_RES),
            AspectBucket(832, 1216, self._TRAIN_RES),
            AspectBucket(832, 1152, self._TRAIN_RES),
            AspectBucket(896, 1152, self._TRAIN_RES),
            AspectBucket(896, 1088, self._TRAIN_RES),
            AspectBucket(960, 1088, self._TRAIN_RES),
            AspectBucket(960, 1024, self._TRAIN_RES),
            # square res bucket
            AspectBucket(1024, 1024, self._TRAIN_RES),
            AspectBucket(1024, 960, self._TRAIN_RES),
            AspectBucket(1088, 960, self._TRAIN_RES),
            AspectBucket(1088, 896, self._TRAIN_RES),
            AspectBucket(1152, 896, self._TRAIN_RES),
            AspectBucket(1152, 832, self._TRAIN_RES),
            AspectBucket(1216, 832, self._TRAIN_RES),
            AspectBucket(1280, 768, self._TRAIN_RES),
            AspectBucket(1344, 768, self._TRAIN_RES),
            AspectBucket(1408, 704, self._TRAIN_RES),
            AspectBucket(1472, 704, self._TRAIN_RES),
            AspectBucket(1536, 640, self._TRAIN_RES),
            AspectBucket(1600, 640, self._TRAIN_RES),
            AspectBucket(1664, 576, self._TRAIN_RES),
            AspectBucket(1728, 576, self._TRAIN_RES),
            AspectBucket(1792, 576, self._TRAIN_RES),
            AspectBucket(1856, 512, self._TRAIN_RES),
            AspectBucket(1920, 512, self._TRAIN_RES),
            AspectBucket(1984, 512, self._TRAIN_RES),
            AspectBucket(2048, 512, self._TRAIN_RES),
        ]
        super().__init__(
            n_buckets=len(self.data),
            edge_min=512,
            edge_max=2048,
            edge_step=64,
            max_aspect=4.0,
            tgt_pixels=self._TRAIN_RES**2,
            tolerance=5,
            bias_square=True,
        )


class WDXLBucketList(AspectBucketList):
    """Hard-coded bucket list matching original WDXL training configuration"""

    _TRAIN_RES = 1024

    def __init__(self):
        self.data: list[AspectBucket] = [
            AspectBucket(512, 2048, self._TRAIN_RES),
            AspectBucket(512, 1984, self._TRAIN_RES),
            AspectBucket(576, 1920, self._TRAIN_RES),
            AspectBucket(576, 1792, self._TRAIN_RES),
            AspectBucket(576, 1728, self._TRAIN_RES),
            # mahouko
            AspectBucket(704, 1472, self._TRAIN_RES),
            AspectBucket(768, 1408, self._TRAIN_RES),
            AspectBucket(768, 1344, self._TRAIN_RES),
            AspectBucket(832, 1280, self._TRAIN_RES),
            AspectBucket(896, 1216, self._TRAIN_RES),
            AspectBucket(896, 1152, self._TRAIN_RES),
            AspectBucket(960, 1152, self._TRAIN_RES),
            AspectBucket(960, 1088, self._TRAIN_RES),
            AspectBucket(1024, 1024, self._TRAIN_RES),
            AspectBucket(1088, 960, self._TRAIN_RES),
            AspectBucket(1152, 960, self._TRAIN_RES),
            AspectBucket(1152, 896, self._TRAIN_RES),
            AspectBucket(1216, 896, self._TRAIN_RES),
            AspectBucket(1280, 832, self._TRAIN_RES),
            AspectBucket(1344, 768, self._TRAIN_RES),
            AspectBucket(1408, 768, self._TRAIN_RES),
            AspectBucket(1472, 704, self._TRAIN_RES),
        ]
        super().__init__(
            n_buckets=len(self.data),
            edge_min=512,
            edge_max=2048,
            edge_step=64,
            max_aspect=4.0,
            tgt_pixels=self._TRAIN_RES**2,
            tolerance=5,
            bias_square=True,
        )
