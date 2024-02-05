import logging

from torch import Tensor, fx
from torch.nn import functional as F
from torchvision import models as tvm
from torchvision.models.feature_extraction import create_feature_extractor

logger = logging.getLogger(__name__)


def create_vgg_extractor(
    features: dict[str, str],
    weights: tvm.VGG16_Weights = tvm.VGG16_Weights.DEFAULT,
    requires_grad: bool = False,
) -> fx.GraphModule:
    model = tvm.vgg16(weights=weights)
    if not requires_grad:
        model = model.eval().requires_grad_(requires_grad)
    return create_feature_extractor(model, features)


def create_alexnet_extractor(
    features: dict[str, str],
    weights: tvm.AlexNet_Weights = tvm.AlexNet_Weights.DEFAULT,
    requires_grad: bool = False,
) -> fx.GraphModule:
    model = tvm.alexnet(weights=weights)
    if not requires_grad:
        model = model.eval().requires_grad_(requires_grad)
    return create_feature_extractor(model, features)


def normalize_tensor(in_feat: Tensor, eps: float = 1e-10) -> Tensor:
    norm_factor = in_feat.pow(2).sum(dim=1, keepdim=True).sqrt()
    return in_feat / (norm_factor + eps)


def spatial_average(x: Tensor, keepdim: bool = True) -> Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def upsample(x: Tensor, out_HW=(64, 64)) -> Tensor:
    return F.upsample(x, size=out_HW, mode="bilinear", align_corners=False)
