"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""
import logging
from collections import OrderedDict, namedtuple

import torch
from torch import Tensor, nn
from torchvision.models import vgg

from neurosis.data import lpips_checkpoint

VggOutputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])

logger = logging.getLogger(__name__)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout: bool = True) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vgg16 features
        self.depth = len(self.chns)
        self.net = vgg16(requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        self._load_pretrained()
        self.requires_grad_(False)

    def _load_pretrained(self, name="vgg_lpips"):
        with lpips_checkpoint(name) as ckpt:
            self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        logger.info("loaded pretrained LPIPS loss from {}.pth".format(name))

    def forward(self, inputs: Tensor, target: Tensor):
        inputs, target = self.scaling_layer(inputs), self.scaling_layer(target)
        inputs, target = self.net(inputs), self.net(target)

        vals = []
        for i in range(self.depth):
            x, y = normalize_tensor(inputs[i]), normalize_tensor(target[i])
            diff = (x - y) ** 2
            z = self.lins[i](diff)
            z = spatial_average(z, keepdim=True)
            vals.append(z)

        return torch.cat(vals).sum()


class ScalingLayer(nn.Module):
    shift: Tensor
    scale: Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp: Tensor) -> Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        self.dropout = nn.Dropout() if use_dropout else None
        self.layer = nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return self.layer(x)


class vgg16(nn.Module):
    def __init__(
        self,
        requires_grad=False,
        weights=vgg.VGG16_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        vgg_pretrained_features = vgg.vgg16(weights=weights).features

        self.slice1 = nn.Sequential(
            OrderedDict([(str(x), vgg_pretrained_features[x]) for x in range(0, 4)]),
        )
        self.slice2 = nn.Sequential(
            OrderedDict([(str(x), vgg_pretrained_features[x]) for x in range(4, 9)]),
        )
        self.slice3 = nn.Sequential(
            OrderedDict([(str(x), vgg_pretrained_features[x]) for x in range(9, 16)]),
        )
        self.slice4 = nn.Sequential(
            OrderedDict([(str(x), vgg_pretrained_features[x]) for x in range(16, 23)]),
        )
        self.slice5 = nn.Sequential(
            OrderedDict([(str(x), vgg_pretrained_features[x]) for x in range(23, 30)]),
        )

        # freeze parameters if not training
        self.requires_grad_(requires_grad)

    def forward(self, x) -> VggOutputs:
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        out = VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x: Tensor, eps=1e-10) -> Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: Tensor, keepdim=True) -> Tensor:
    return x.mean([2, 3], keepdim=keepdim)
