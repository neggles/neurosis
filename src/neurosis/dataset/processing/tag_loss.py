import logging
from bisect import bisect_left
from collections import UserDict
from typing import Callable, Optional

import lightning.pytorch as L
import numpy as np
import torch
from torch import Tensor

from neurosis.modules.diffusion.hooks import LossHook
from neurosis.utils import np_text_decode

logger = logging.getLogger(__name__)

DEFAULT_SCALE = {
    -1: 1.1,
    10: 1.05,
    50: 1.02,
    100: 1,
    1000: 0.999,
    2000: 0.995,
    4000: 0.99,
    6000: 0.98,
    8000: 0.97,
    10000: 0.96,
    15000: 0.95,
    20000: 0.90,
    30000: 0.85,
    40000: 0.80,
}


class TagFreqScale(UserDict):
    data: dict[int, float]
    steps: list[int]

    def __init__(
        self,
        scales: list[tuple[int, float]] | dict[int, float] = DEFAULT_SCALE,
    ):
        if isinstance(scales, list):
            scales = dict(scales)
        super().__init__(data=scales)
        self.steps = sorted(self.keys())

    def __getitem__(self, key: int):
        if key not in self.data:
            key = self.steps[bisect_left(self.steps, key)]
        return self.data[key]

    def __setitem__(self, key: int, value: float):
        ret = super().__setitem__(key, value)
        self.steps = sorted(self.keys())
        return ret


class TagRewards(UserDict):
    data: dict[str, float]

    def __init__(
        self,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if isinstance(v, float)}
        super().__init__(data=kwargs)

    def __getitem__(self, key: str):
        if key not in self:
            return None


class TagCount(UserDict):
    data: dict[str, int]

    def __getitem__(self, key):
        if key not in self:
            self[key] = 0
        return super().__getitem__(key)

    def increment(self, tag: str) -> int:
        if tag in self:
            self[tag] += 1
        else:
            self[tag] = 1
        return self[tag]

    def reset(self, tag: Optional[str] = None):
        if tag is not None:
            self[tag] = 0
        else:
            self.data: dict[str, int] = {}


class TagLoss(UserDict):
    data: dict[str, tuple[float, int]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TagFrequencyHook(LossHook):
    def __init__(
        self,
        # function to check if a tag is an artist or character tag
        check_fn: Callable,
        # input key for caption text
        input_key: str = "caption",
        # separator for tags in tag string
        tag_sep: str = ", ",
        # strength of loss pull towards historical frequency adjusted tag loss
        alpha: float = 0.2,
        # strength of historical loss accumulation
        beta: float = 0.99,
        # overall strength of tag loss
        strength: float = 1.0,
        # loss multipliers for tags based on frequency of occurrence
        freq_scale: TagFreqScale = TagFreqScale(),
        # specific adjustments for individual tags
        tag_rewards: TagRewards = TagRewards(),
    ):
        self.input_key = input_key
        self.check_fn = check_fn

        self.alpha = alpha
        self.beta = beta
        self.strength = strength
        self.freq_scale = freq_scale if isinstance(freq_scale, TagFreqScale) else TagFreqScale(freq_scale)
        self.tag_rewards = tag_rewards if isinstance(tag_rewards, TagRewards) else TagRewards(tag_rewards)
        self.tag_sep = tag_sep

        self.tag_counter: TagCount = TagCount()
        self.loss_stats: TagLoss = TagLoss()

        # used for the current batch between pre-hook and batch-hook
        self.epoch = 0
        self.global_step = 0
        self.batch_idx = 0
        self.tags: list[str] = None
        self.total_loss: float = 0.0
        self.ucg_batch: bool = False

    def pre_hook(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx,
    ):
        if self.tags is not None:
            raise RuntimeError("pre_hook called twice without calling batch_hook")

        self.global_step = trainer.global_step
        self.batch_idx = batch_idx
        self.tags = [np_text_decode(x) for x in batch[self.input_key]]
        self.ucg_batch = batch.get("is_ucg", False)

        if pl_module.current_epoch != self.epoch:
            self.epoch = pl_module.current_epoch
            self.tag_counter.reset()

    def get_batch_tags(self, count: int) -> list[str]:
        tags = self.tags[:count]
        self.tags = self.tags[count:]
        if len(self.tags) == 0:
            self.tags = None
        return tags

    def batch_hook(
        self,
        pl_module: L.LightningModule,
        batch: dict,
        loss: Tensor,
        loss_dict: dict[str, Tensor] = {},
        **kwargs,
    ):
        if not self.tags:
            raise RuntimeError("get_weight called without calling save_tags")

        batch_len = loss.shape[0]
        tags = self.get_batch_tags(batch_len)

        base_loss = [x.detach().mean().cpu().item() for x in loss]
        base_acc = sum(base_loss) / batch_len

        if self.total_loss <= 0.0:
            self.total_loss = base_acc
        else:
            # first 10 samples get more influence to warm up the stats
            batch_beta = min(self.beta, self.global_step / 10.0)
            self.total_loss = (self.total_loss * batch_beta) + (base_acc * (1.0 - batch_beta))

        weights = []
        for i in range(batch_len):
            base_mult = 1

            sample_tags = [x for x in tags[i].split(self.tag_sep)]
            sample_loss = base_loss[i]
            tag_mults = []
            base_mults = []

            if not self.ucg_batch:
                adjust_tags = list(filter(self.check_fn, sample_tags))
            else:
                adjust_tags = []

            for tag in adjust_tags:
                count = self.tag_counter.increment(tag)
                base_mults.append(self.freq_scale[count])

                if tag in self.loss_stats:
                    tag_loss, tag_count = self.loss_stats[tag]
                    tag_beta = min(self.beta, tag_count / 10.0)
                    self.loss_stats[tag] = (tag_loss * tag_beta) + (sample_loss * (1.0 - tag_beta))
                else:
                    self.loss_stats[tag] = (sample_loss, 1.0)
                    tag_loss = sample_loss
                tag_mults.append(tag_loss)

            # apply tag rewards to loss to make images with desirable tags be learned more
            for tag in [x for x in sample_tags if x in self.tag_rewards]:
                base_mult *= self.tag_rewards[tag]
            # apply frequency adjust multiplier
            if len(base_mults) > 0:
                base_mult *= np.array(base_mults).mean()

            # grab the historical rolling average loss for these tags
            hist_loss = np.array(tag_mults).mean() if len(tag_mults) > 0 else sample_loss

            # pull current batch item loss towards hist_loss with alpha strength
            target_loss = (sample_loss * (1.0 - self.alpha)) + (hist_loss * self.alpha)
            # apply rewards/punishments for frequency and good/bad tags
            target_loss *= base_mult
            # get ratio of adjusted loss to rolling average loss
            loss_weight = target_loss / base_acc
            # adjust for global modifier strength
            loss_weight = 1.0 + self.strength * (loss_weight - 1.0)

            weights.append(torch.ones(loss.shape[1:]) * loss_weight)

        loss_dict.update({"train/frequency_weight": torch.stack(weights).detach().mean().cpu()})
        weights = torch.stack(weights).to(loss)

        return loss * weights, loss_dict


## tag classifier functions
def is_artist_or_character(tag: str) -> bool:
    """Naive check for artist or character tags, requires tag to start with 'artist:' or 'character:'"""
    return tag.startswith("character:") or tag.startswith("artist:")
