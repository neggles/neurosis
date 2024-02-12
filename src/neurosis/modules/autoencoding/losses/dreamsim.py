import logging

from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.modules.losses.perceptual import LPIPS
from neurosis.modules.losses.types import GenericLoss, PerceptualLoss
from neurosis.trainer.util import EMATracker

logger = logging.getLogger(__name__)


class AutoencoderPerceptual(nn.Module):
    def __init__(
        self,
        recon_type: GenericLoss | str = GenericLoss.L1,
        recon_weight: float = 1.0,
        perceptual_type: PerceptualLoss | str = PerceptualLoss.LPIPS,
        perceptual_weight: float = 1.0,
        lpips_type: str = "alex",
        resize_input: bool = False,
        resize_target: bool = False,
        loss_ema_steps: int = 64,
        extra_log_keys: list[str] = [],
    ):
        super().__init__()
        # set up reconstruction loss
        self.recon_type = recon_type
        match self.recon_type:
            case GenericLoss.L1:
                self.recon_loss = nn.L1Loss(reduction="none")
            case GenericLoss.L2 | GenericLoss.MSE:
                self.recon_loss = nn.MSELoss(reduction="none")
            case _:
                raise ValueError(f"Unknown reconstruction loss type {self.recon_type}")
        self.recon_weight = recon_weight

        # validate the resize options (not likely to be used anyway)
        if resize_input and resize_target:
            raise ValueError("Only one of resize_input and resize_target can be True")
        self.scale_input_to_target = resize_input
        self.scale_target_to_input = resize_target

        # set up perceptual loss
        self.percep_type = PerceptualLoss(perceptual_type.lower())
        if self.percep_type != PerceptualLoss.LPIPS:
            raise NotImplementedError(f"Perceptual loss {perceptual_type} not implemented")
        self.percep_loss = LPIPS(pnet_type=lpips_type).eval()
        self.percep_weight = perceptual_weight

        # set up EMA loss tracking if desired
        self.t_ema = EMATracker(steps=loss_ema_steps)
        self.p_ema = EMATracker(steps=loss_ema_steps)
        self.r_ema = EMATracker(steps=loss_ema_steps)

        # keys to be used in the forward method
        self.forward_keys = ["split"]
        # extra log keys
        self.extra_log_keys = set(extra_log_keys)

    def forward(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        split: str = "train",
        **kwargs,
    ) -> tuple[Tensor, dict]:
        if self.scale_input_to_target:
            inputs = F.interpolate(inputs, reconstructions.shape[2:], mode="bicubic", antialias=True)
        if self.scale_target_to_input:
            reconstructions = F.interpolate(reconstructions, inputs.shape[2:], mode="bicubic", antialias=True)

        # do reconstruction and perceptual loss
        inputs = inputs.clamp(-1.0, 1.0).contiguous()
        reconstructions = reconstructions.clamp(-1.0, 1.0).contiguous()

        rec_loss = self.recon_loss(inputs, reconstructions)
        p_loss = self.percep_loss(inputs, reconstructions).relu()
        loss = (rec_loss * self.recon_weight) + (p_loss * self.percep_weight)

        log_loss = loss.detach().clone().mean()
        log_rec_loss = rec_loss.detach().mean()
        log_p_loss = p_loss.detach().mean()

        self.t_ema.update(log_loss.item())
        self.r_ema.update(log_rec_loss.item())
        self.p_ema.update(log_p_loss.item())

        log_dict = {
            f"{split}/loss/total": log_loss,
            f"{split}/loss/rec": log_rec_loss,
            f"{split}/loss/p": log_p_loss,
            f"{split}/loss/total_ema": self.t_ema.value,
            f"{split}/loss/rec_ema": self.r_ema.value,
            f"{split}/loss/p_ema": self.p_ema.value,
        }
        return loss, log_dict
