from enum import Enum


class PerceptualLoss(str, Enum):
    LPIPS = "lpips"
    MSE = "mse"
    DISTS = "dists"


class DreamsimVariant(str, Enum):
    DinoB16 = "dino_vitb16"
    ClipB32 = "clip_vitb32"
    OpenClipB32 = "open_clip_vitb32"
    EnsembleB16 = "ensemble_vitb16"


class GenericLoss(str, Enum):
    L1 = "l1"
    L2 = "l2"
    MSE = "mse"
    NLL = "nll"


class DiscriminatorLoss(str, Enum):
    Vanilla = "vanilla"
    Hinge = "hinge"


class DiffusionObjective(str, Enum):
    EDM = "edm"
    RF = "rf"
