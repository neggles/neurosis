from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.modules.diffusion.model import Decoder
from neurosis.modules.losses.perceptual import LPIPS


class LatentLPIPS(nn.Module):
    def __init__(
        self,
        decoder: Decoder,
        perceptual_weight: float = 1.0,
        latent_weight: float = 1.0,
        scale_input_to_tgt_size: bool = False,
        scale_tgt_to_input_size: bool = False,
        perceptual_weight_on_inputs: float = 0.0,
    ):
        super().__init__()
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        self.scale_tgt_to_input_size = scale_tgt_to_input_size
        self.decoder = decoder
        if hasattr(self.decoder, "encoder"):
            delattr(self.decoder, "encoder")

        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.latent_weight = latent_weight
        self.perceptual_weight_on_inputs = perceptual_weight_on_inputs

    def forward(
        self,
        latent_inputs: Tensor,
        latent_predictions: Tensor,
        image_inputs: Tensor,
        split="train",
    ) -> tuple[Tensor, dict[str, Any]]:
        log = dict()
        loss = (latent_inputs - latent_predictions) ** 2
        log[f"{split}/latent_l2_loss"] = loss.mean().detach()
        image_reconstructions: Tensor = None
        if self.perceptual_weight > 0.0:
            image_reconstructions = self.decoder.decode(latent_predictions)
            image_targets = self.decoder.decode(latent_inputs)
            perceptual_loss = self.perceptual_loss(
                image_targets.contiguous(), image_reconstructions.contiguous()
            )
            loss = self.latent_weight * loss.mean() + self.perceptual_weight * perceptual_loss.mean()
            log[f"{split}/perceptual_loss"] = perceptual_loss.mean().detach()

        if self.perceptual_weight_on_inputs > 0.0:
            image_reconstructions = image_reconstructions or self.decoder.decode(latent_predictions)
            if self.scale_input_to_tgt_size:
                image_inputs = F.interpolate(
                    image_inputs,
                    image_reconstructions.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )
            elif self.scale_tgt_to_input_size:
                image_reconstructions = F.interpolate(
                    image_reconstructions,
                    image_inputs.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )

            perceptual_loss2 = self.perceptual_loss(
                image_inputs.contiguous(), image_reconstructions.contiguous()
            )
            loss = loss + self.perceptual_weight_on_inputs * perceptual_loss2.mean()
            log[f"{split}/perceptual_loss_on_inputs"] = perceptual_loss2.mean().detach()
        return loss, log
