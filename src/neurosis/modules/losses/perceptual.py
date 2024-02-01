"""Stripped and reworked version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""
import logging
import warnings
from collections import namedtuple

from torch import Tensor, fx, nn
from torch.nn import functional as F
from torchvision import models as tvm
from torchvision.models.feature_extraction import create_feature_extractor

from neurosis.data import lpips_checkpoint

VggOutputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=r"^'has_.*' is deprecated")


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(
        self,
        pretrained=True,
        lpips: bool = True,
        pnet_rand: bool = False,
        pnet_tune: bool = False,
        use_dropout: bool = True,
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
        if verbose:
            print("Setting up [LPIPS] perceptual loss: trunk [vgg16], v[0.1]")

        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.scaling_layer = ScalingLayer()

        self.chns = [64, 128, 256, 512, 512]

        pnet_feats = {
            "features.3": "relu1_2",
            "features.8": "relu2_2",
            "features.15": "relu3_3",
            "features.22": "relu4_3",
            "features.29": "relu5_3",
        }
        self.pnet_keys = list(pnet_feats.values())
        self.pnet: fx.GraphModule = create_vgg_extractor(
            features=pnet_feats,
            weights=tvm.VGG16_Weights.IMAGENET1K_V1 if not pnet_rand else None,
            requires_grad=pnet_tune,
        )

        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            self.lins = nn.ModuleDict({key: val for key, val in zip(self.pnet_keys, self.lins)})

            if pretrained:
                self._load_pretrained()

        if freeze:
            self.requires_grad_(False)

    def _load_pretrained(self, name="vgg_lpips_v0.1"):
        with lpips_checkpoint(name) as state_dict:
            self.load_state_dict(state_dict, strict=False)
        logger.info("loaded pretrained LPIPS loss from {}.pth".format(name))

    def forward(self, x: Tensor, y: Tensor, retPerLayer: bool = False, normalize: bool = False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            x = x.mul(2.0).add(-1.0)
            y = y.mul(2.0).add(-1.0)

        inX, inY = self.scaling_layer(x), self.scaling_layer(y)
        outX, outY = self.pnet.forward(inX), self.pnet.forward(inY)

        res = {}
        val = 0
        for key in self.pnet_keys:
            featsX, featsY = normalize_tensor(outX[key]), normalize_tensor(outY[key])
            diffs = featsX.sub(featsY).pow(2)
            if self.lpips:
                diffs = self.lins[key](diffs)
                res[key] = spatial_average(diffs, keepdim=True)
            else:
                diffs = diffs.sum(dim=1, keepdim=True)
                res[key] = upsample(diffs, out_HW=x.shape[2:])
            val += res[key]

        if retPerLayer:
            return (val, VggOutputs(**res))
        return val


class ScalingLayer(nn.Module):
    shift: Tensor
    scale: Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("shift", Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, x: Tensor) -> Tensor:
        x = x.sub(self.shift).div(self.scale)
        return x


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        self.dropout = nn.Dropout() if use_dropout else nn.Identity()
        self.layer = nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.layer(x)
        return x


def create_vgg_extractor(
    features: dict[str, str],
    weights: tvm.VGG16_Weights = tvm.VGG16_Weights.IMAGENET1K_V1,
    requires_grad: bool = False,
) -> fx.GraphModule:
    vgg = tvm.vgg16(weights=weights)
    if not requires_grad:
        vgg = vgg.eval()
        vgg.requires_grad_(requires_grad)
    return create_feature_extractor(vgg, features)


def normalize_tensor(in_feat: Tensor, eps: float = 1e-10) -> Tensor:
    norm_factor = in_feat.pow(2).sum(dim=1, keepdim=True).sqrt()
    return in_feat / (norm_factor + eps)


def spatial_average(x: Tensor, keepdim: bool = True) -> Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def upsample(x: Tensor, out_HW=(64, 64)) -> Tensor:
    return F.upsample(x, size=out_HW, mode="bilinear", align_corners=False)
