from enum import Enum


class PerceptualLoss(str, Enum):
    LPIPS = "lpips"
    MSE = "mse"
    DISTS = "dists"
    DreamSim = "dreamsim"


class GenericLoss(str, Enum):
    L1 = "l1"
    L2 = "l2"
    MSE = "mse"
    NLL = "nll"


class DiscriminatorLoss(str, Enum):
    Vanilla = "vanilla"
    Hinge = "hinge"
