import torch
from torch import Tensor
from torch.nn import functional as F


def adopt_weight(weight: Tensor, global_step: int, threshold: int = 0, value=0.0) -> Tensor:
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = torch.mul(0.5, torch.add(loss_real, loss_fake))
    return d_loss


def vanilla_d_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = torch.mul(0.5, torch.add(loss_real, loss_fake))
    return d_loss
