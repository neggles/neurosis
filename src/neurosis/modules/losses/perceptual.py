"""Stripped and reworked version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import logging
import warnings
from typing import NamedTuple

import torch
from torch import Tensor, fx, nn
from torch.nn import functional as F

from neurosis.data import lpips_checkpoint
from neurosis.modules.losses.extractors import create_alexnet_extractor, create_vgg_extractor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=r"^'has_.*' is deprecated")


class AlexOutputs(NamedTuple):
    relu1: Tensor
    relu2: Tensor
    relu3: Tensor
    relu4: Tensor
    relu5: Tensor


class VggOutputs(NamedTuple):
    relu1: Tensor
    relu2: Tensor
    relu3: Tensor
    relu4: Tensor
    relu5: Tensor


PNET_CONFIG = {
    "alex": {
        "channels": [64, 192, 384, 256, 256],
        "features": {
            "features.1": "relu1",
            "features.4": "relu2",
            "features.7": "relu3",
            "features.9": "relu4",
            "features.11": "relu5",
        },
        "load_fn": create_alexnet_extractor,
        "out_type": AlexOutputs,
    },
    "vgg": {
        "channels": [64, 128, 256, 512, 512],
        "features": {
            "features.3": "relu1",
            "features.8": "relu2",
            "features.15": "relu3",
            "features.22": "relu4",
            "features.29": "relu5",
        },
        "load_fn": create_vgg_extractor,
        "out_type": VggOutputs,
    },
}


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(
        self,
        pnet_type: str = "alex",
        pretrained: bool = True,
        lpips: bool = True,
        pnet_rand: bool = False,
        pnet_tune: bool = False,
        use_dropout: bool = False,
        spatial: bool = False,
        freeze: bool = True,
        verbose: bool = True,
    ):
        """Initializes a perceptual loss nn.Module

        Parameters (default listed first)
        ---------------------------------
        lpips : bool
            [True] use linear layers on top of base/trunk network
            [False] means no linear layers; each layer is averaged together
        pretrained : bool
            This flag controls the linear layers, which are only in effect when lpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk

        The following parameters should only be changed if training the network

        freeze : bool
            [True] is for test mode (default)
            [False] is for training mode
        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        """

        super().__init__()
        if "vgg" in pnet_type:
            pnet_type = "vgg"
        if "alex" in pnet_type:
            pnet_type = "alex"

        if verbose:
            print(f"Setting up [LPIPS] perceptual loss: trunk [{pnet_type}], v[0.1]")

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.pnet_conf = PNET_CONFIG[pnet_type]
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.spatial = spatial

        self.scaling_layer = ScalingLayer()
        self.chns = self.pnet_conf["channels"]
        self.L = len(self.chns)
        self.out_type: AlexOutputs | VggOutputs = self.pnet_conf["out_type"]

        pnet_feats = self.pnet_conf["features"]
        self.pnet_keys = list(pnet_feats.values())
        self.pnet: fx.GraphModule = self.pnet_conf["load_fn"](features=pnet_feats, requires_grad=pnet_tune)

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            self.lins = nn.ModuleDict({key: val for key, val in zip(self.pnet_keys, lins)})

            if pretrained:
                self._load_pretrained(pnet_type)

        if freeze:
            self.requires_grad_(False)

    @property
    def device(self):
        return self.parameters().__next__().device

    @property
    def dtype(self):
        return self.parameters().__next__().dtype

    def _load_pretrained(self, name: str):
        with lpips_checkpoint(name) as state_dict:
            self.load_state_dict(state_dict, strict=False)
        logger.info(f"loaded pretrained LPIPS loss from '{name}'")

    def forward(self, x: Tensor, y: Tensor, retPerLayer: bool = False, normalize: bool = False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            x, y = x.mul(2.0).add(-1.0), y.mul(2.0).add(-1.0)

        inX, inY = self.scaling_layer(x), self.scaling_layer(y)
        outX, outY = self.pnet.forward(inX), self.pnet.forward(inY)

        res = {}
        val = 0
        for key in self.pnet_keys:
            featsX, featsY = normalize_tensor(outX[key]), normalize_tensor(outY[key])
            diffs = featsX.sub(featsY).pow(2)
            if self.lpips:
                diffs = self.lins[key](diffs)
                if self.spatial:
                    res[key] = upsample(diffs, out_HW=x.shape[2:])
                else:
                    res[key] = spatial_average(diffs, keepdim=True)
            else:
                diffs = diffs.sum(dim=1, keepdim=True)
                if self.spatial:
                    res[key] = upsample(diffs, out_HW=x.shape[2:])
                else:
                    res[key] = spatial_average(diffs, keepdim=True)
            val += res[key]

        if retPerLayer:
            return (val, self.out_type(**res))
        return val


class ScalingLayer(nn.Module):
    shift: Tensor
    scale: Tensor

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def normalize_tensor(in_feat: Tensor, eps: float = 1e-10) -> Tensor:
    norm_factor = in_feat.pow(2).sum(dim=1, keepdim=True).sqrt()
    return in_feat / (norm_factor + eps)


def spatial_average(x: Tensor, keepdim: bool = True) -> Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def upsample(x: Tensor, out_HW=(64, 64)) -> Tensor:
    return F.upsample(x, size=out_HW, mode="bilinear", align_corners=False)
