from collections import UserDict
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from neurosis.modules.losses.hooks import LossHook

FREQ_SCALES = {
    "NAI": {
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
}

FREQ_REWARDS = {
    "NAI": {
        "bad_anatomy": 0.99,
        "bad_feet": 0.99,
        "bad_hands": 0.99,
        "bad_leg": 0.99,
        "best_quality": 1.015,
        "censored": 0.975,
        "comic": 0.99,
        "error": 0.98,
        "everyone": 1.0025,
        "jpeg_artifacts": 0.99,
        "lowres": 0.99,
        "masterpiece": 1.03,
        "sample_watermark": 0.95,
        "scenery": 1.005,
        "uncensored": 1.01,
    }
}


def is_artist_or_character(tag):
    return tag.startswith("character:") or tag.startswith("artist:")


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
        steps_per_epoch: int,
        # function to check if a tag is an artist or character tag
        tag_check_fn=is_artist_or_character,
        # strength of loss pull towards historical frequency adjusted tag loss
        alpha: float = 0.2,
        # strength of historical loss accumulation
        beta: float = 0.99,
        # overall strength of tag loss
        strength: float = 1.0,
        # loss multipliers for tags based on frequency of occurrence
        freq_scale: dict[int, float] = FREQ_SCALES["NAI"],
        # specific adjustments for individual tags
        freq_rewards: dict[str, float] = FREQ_REWARDS["NAI"],
        # separator for tags in tag string
        tag_sep: str = ", ",
    ):
        self.steps_per_epoch = steps_per_epoch
        self.global_step = 0

        self.alpha = alpha
        self.beta = beta
        self.strength = strength
        self.freq_scale = freq_scale
        self.tag_rewards = freq_rewards
        self.tag_sep = tag_sep

        self.check_tag = tag_check_fn

        # mapping from tag to count of how many times it has been seen in the current epoch
        self.tag_counter: TagCount = TagCount()
        self.loss_stats: TagLoss = TagLoss()

        self.registered: bool = False
        self.tags: list[str] = None
        self.total_loss: float = 0.0

        self.ucg_batch: bool = False

    @property
    def freq_steps(self):
        return sorted(self.freq_scale.keys())

    def save_batch_tags(self, tags: list[str], is_ucg: bool = False):
        if self.tags is not None:
            raise RuntimeError("save_tags called twice without calling get_weight")

        self.global_step += 1
        self.tags = tags
        self.ucg_batch = is_ucg

        if not (self.global_step % self.steps_per_epoch):
            self.tag_counter.reset()

    def get_batch_tags(self, count: int) -> list[str]:
        tags = self.tags[:count]
        self.tags = self.tags[count:]
        if len(self.tags) == 0:
            self.tags = None
        return tags

    def get_scale(self, count: int) -> float:
        step = self.freq_steps[np.searchsorted(self.freq_steps, count, "left")]
        return self.freq_scale[step]

    def get_weight(self, loss: Tensor):
        if not self.tags:
            raise RuntimeError("get_weight called without calling save_tags")

        batch_len = loss.shape[0]
        tags = self.get_batch_tags(batch_len)

        loss_simple = [x.detach().mean().cpu().item() for x in loss]
        acc_simple = sum(loss_simple) / batch_len

        if self.total_loss <= 0.0:
            self.total_loss = acc_simple
        else:
            # first 10 samples get more influence to warm up the stats
            batch_beta = min(self.beta, self.global_step / 10.0)
            self.total_loss = (self.total_loss * batch_beta) + (acc_simple * (1.0 - batch_beta))

        weights = []
        for i in range(batch_len):
            base_mult = 1

            sample_tags = [x for x in tags[i].split(self.tag_sep)]
            if not self.ucg_batch:
                adjust_tags = list(filter(self.check_tag, sample_tags))
            else:
                adjust_tags = []

            sample_loss = loss_simple[i]
            tag_mults = []
            base_mults = []
            for tag in adjust_tags:
                count = self.tag_counter.increment(tag)
                base_mults.append(self.get_scale(count))

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
            loss_weight = target_loss / acc_simple
            # adjust for global modifier strength
            loss_weight = 1.0 + self.strength * (loss_weight - 1.0)

            weights.append(torch.ones(loss.shape[1:]) * loss_weight)

        weights = torch.stack(weights).to(loss.device)

        return weights
