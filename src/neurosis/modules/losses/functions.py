import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.modules.losses.types import DiscriminatorLoss


def apply_threshold_weight(
    value: float | Tensor,
    global_step: int,
    start_step: int = 0,
    weight: float | Tensor = 1.0,
) -> Tensor:
    """Return a weighted value if the global_step is greater than the start global_step, else 0.0."""
    global_step = torch.tensor(global_step)
    return torch.where(torch.lt(global_step, start_step), 0.0, weight.mul(value))


class HingeDiscLoss(nn.Module):
    def __init__(self, weight: float = 1.0, start_step: int = 0):
        super().__init__()
        self.weight = weight
        self.start_step = start_step

    def forward(self, real: Tensor, fake: Tensor, global_step: int = -1) -> Tensor:
        if self.start_step > 0 and global_step < self.start_step:
            return torch.zeros(1, device=real.device)
        loss_real = F.relu(1.0 - real).mean()
        loss_fake = F.relu(1.0 + fake).mean()
        d_loss = loss_real.add(loss_fake).mul(0.5)
        return d_loss.mul(self.weight)


class VanillaDiscLoss(nn.Module):
    def __init__(self, weight: float = 1.0, start_step: int = 0):
        super().__init__()
        self.weight = weight
        self.start_step = start_step

    def forward(self, real: Tensor, fake: Tensor, global_step: int = -1) -> Tensor:
        if self.start_step > 0 and global_step < self.start_step:
            return torch.zeros(1, device=real.device)
        loss_real = F.softplus(-real).mean()
        loss_fake = F.softplus(fake).mean()
        d_loss = loss_real.add(loss_fake).mul(0.5)
        return d_loss.mul(self.weight)


def get_discr_loss_fn(
    kind: DiscriminatorLoss = DiscriminatorLoss.Hinge,
    weight: float = 1.0,
    start_step: int = 0,
) -> HingeDiscLoss | VanillaDiscLoss:
    match kind:
        case "hinge":
            return HingeDiscLoss(weight, start_step)
        case "vanilla":
            return VanillaDiscLoss(weight, start_step)
        case _:
            raise ValueError(f"Unknown discriminator loss: {kind}")
