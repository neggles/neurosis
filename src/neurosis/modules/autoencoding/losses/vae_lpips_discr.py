import logging
from typing import Iterator, Optional

import numpy as np
import torch
from einops import rearrange
from matplotlib import (
    colormaps,
    pyplot as plt,
)
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.utils import make_grid

from neurosis.modules.losses.functions import get_discr_loss_fn
from neurosis.modules.losses.patchgan import NLayerDiscriminator, weights_init
from neurosis.modules.losses.perceptual import LPIPS
from neurosis.modules.losses.types import DiscriminatorLoss, GenericLoss, PerceptualLoss

logger = logging.getLogger(__name__)


class AutoencoderLPIPSWithDiscr(nn.Module):
    def __init__(
        self,
        recon_type: GenericLoss | str = GenericLoss.L1,
        recon_weight: float = 1.0,
        perceptual_type: PerceptualLoss | str = PerceptualLoss.LPIPS,
        perceptual_weight: float = 1.0,
        disc_start: int = 0,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        disc_lambda_r1: float = 0.0,
        disc_loss: DiscriminatorLoss | str = DiscriminatorLoss.Hinge,
        disc_kwargs: Optional[dict] = {},
        resize_input: bool = False,
        resize_target: bool = False,
        extra_log_keys: list[str] = [],
    ):
        super().__init__()

        self.recon_type = recon_type
        match self.recon_type:
            case GenericLoss.L1:
                self.recon_loss = nn.L1Loss(reduction="none")
            case GenericLoss.L2 | GenericLoss.MSE:
                self.recon_loss = nn.MSELoss(reduction="none")
            case _:
                raise ValueError(f"Unknown reconstruction loss type {self.recon_type}")
        self.recon_weight = recon_weight

        if perceptual_type != PerceptualLoss.LPIPS:
            raise NotImplementedError(f"Perceptual loss {perceptual_type} not implemented")
        self.percep_loss = LPIPS().eval()
        self.percep_weight = perceptual_weight

        self.disc_start = disc_start
        self.disc_factor = disc_factor
        self.disc_weight = disc_weight
        self.disc_lambda_r1 = disc_lambda_r1

        disc_config = dict(input_nc=3, n_layers=3, use_actnorm=False)
        if disc_kwargs is not None:
            disc_config.update(disc_kwargs)
        self.discr = NLayerDiscriminator(**disc_config).apply(weights_init)
        self.discr_loss = get_discr_loss_fn(disc_loss, disc_weight, disc_start)

        if resize_input and resize_target:
            raise ValueError("Only one of resize_input and resize_target can be True")
        self.scale_input_to_target = resize_input
        self.scale_target_to_input = resize_target

        self.forward_keys = [
            "global_step",
            "optimizer_idx",
            "split",
        ]

        self.extra_log_keys = set(extra_log_keys)

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.discr.parameters()

    @torch.no_grad()
    def log_images(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        split: str = "train",
        **kwargs,
    ) -> dict[str, Tensor]:
        # calc logits of real/fake
        logits_real: Tensor = self.discr(inputs.contiguous().detach())
        if len(logits_real.shape) < 4:
            # Non patch-discriminator
            return dict()
        logits_fake = self.discr(reconstructions.contiguous().detach())
        # -> (b, 1, h, w)

        # parameters for colormapping
        high = max(logits_fake.abs().max(), logits_real.abs().max()).item()
        cmap = colormaps["PiYG"]  # diverging colormap

        def to_colormap(logits: Tensor) -> Tensor:
            """(b, 1, ...) -> (b, 3, ...)"""
            logits = (logits + high) / (2 * high)
            logits_np = cmap(logits.cpu().numpy())[..., :3]  # truncate alpha channel
            # -> (b, 1, ..., 3)
            logits = torch.from_numpy(logits_np).to(logits.device)
            return rearrange(logits, "b 1 ... c -> b c ...")

        logits_real = torch.nn.functional.interpolate(
            logits_real,
            size=inputs.shape[-2:],
            mode="nearest",
            antialias=False,
        )
        logits_fake = torch.nn.functional.interpolate(
            logits_fake,
            size=reconstructions.shape[-2:],
            mode="nearest",
            antialias=False,
        )

        # alpha value of logits for overlay
        alpha_real = torch.abs(logits_real) / high
        alpha_fake = torch.abs(logits_fake) / high
        # -> (b, 1, h, w) in range [0, 0.5]
        # alpha value of lines don't really matter, since the values are the same
        # for both images and logits anyway
        grid_alpha_real = make_grid(alpha_real, nrow=4)
        grid_alpha_fake = make_grid(alpha_fake, nrow=4)
        grid_alpha = 0.8 * torch.cat((grid_alpha_real, grid_alpha_fake), dim=1)
        # -> (1, h, w)
        # blend logits and images together

        # prepare logits for plotting
        logits_real = to_colormap(logits_real)
        logits_fake = to_colormap(logits_fake)
        # resize logits
        # -> (b, 3, h, w)

        # make some grids
        # add all logits to one plot
        logits_real = make_grid(logits_real, nrow=4)
        logits_fake = make_grid(logits_fake, nrow=4)
        # I just love how torchvision calls the number of columns `nrow`
        grid_logits = torch.cat((logits_real, logits_fake), dim=1)
        # -> (3, h, w)

        grid_images_real = make_grid(0.5 * inputs + 0.5, nrow=4)
        grid_images_fake = make_grid(0.5 * reconstructions + 0.5, nrow=4)
        grid_images = torch.cat((grid_images_real, grid_images_fake), dim=1)
        # -> (3, h, w) in range [0, 1]

        grid_blend = grid_alpha * grid_logits + (1 - grid_alpha) * grid_images

        # Create labeled colorbar
        dpi = 100
        height = 128 / dpi
        width = grid_logits.shape[2] / dpi
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        img = ax.imshow(np.array([[-high, high]]), cmap=cmap)
        plt.colorbar(
            img,
            cax=ax,
            orientation="horizontal",
            fraction=0.9,
            aspect=width / height,
            pad=0.0,
        )
        img.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        # manually convert figure to numpy
        cbar_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        cbar_np = cbar_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cbar = torch.from_numpy(cbar_np.copy()).to(grid_logits.dtype) / 255.0
        cbar = rearrange(cbar, "h w c -> c h w").to(grid_logits.device)

        # Add colorbar to plot
        annotated_grid = torch.cat((grid_logits, cbar), dim=1)
        blended_grid = torch.cat((grid_blend, cbar), dim=1)
        return {
            "vis_logits": 2 * annotated_grid[None, ...] - 1,
            "vis_logits_blended": 2 * blended_grid[None, ...] - 1,
        }

    def calc_r1_penalty(self, real: Tensor) -> Tensor:
        real.requires_grad_(True)
        logits = self.discr(real)
        grads = torch.autograd.grad(logits.mean(), real, create_graph=True, retain_graph=True)[0]
        grad_penalty = grads.pow(2).sum(dim=(1, 2, 3)).mean()
        return grad_penalty.mul(self.disc_lambda_r1)

    def forward(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        global_step: int,
        optimizer_idx: int = 0,
        split: str = "train",
    ) -> tuple[Tensor, dict]:
        if self.scale_input_to_target:
            inputs = F.interpolate(inputs, reconstructions.shape[2:], mode="bicubic", antialias=True)
        if self.scale_target_to_input:
            reconstructions = F.interpolate(reconstructions, inputs.shape[2:], mode="bicubic", antialias=True)

        # do reconstruction and perceptual loss
        rec_loss = self.recon_loss(inputs.contiguous(), reconstructions.contiguous())
        if self.percep_weight > 0:
            p_loss = self.percep_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = F.relu(p_loss)
            p_rec_loss = (rec_loss * self.recon_weight) + (p_loss * self.percep_weight)
        else:
            p_loss = torch.tensor(0.0, requires_grad=True)
            p_rec_loss = rec_loss * self.recon_weight

        # now for GAN
        if optimizer_idx == 0:
            # update the generator
            if (global_step < self.disc_start) or (self.training is False):
                r1_penalty = self.calc_r1_penalty(inputs)
                logits_fake: Tensor = self.discr(reconstructions.contiguous())
                g_loss = logits_fake.mean().neg() + r1_penalty
            else:
                r1_penalty = torch.tensor(0.0, requires_grad=True)
                g_loss = torch.tensor(0.0, requires_grad=True)

            g_weighted = g_loss * self.disc_factor
            loss = p_rec_loss + g_weighted

            log_dict = {
                f"{split}/loss/rec": rec_loss.detach().mean(),
                f"{split}/loss/p": p_loss.detach().mean(),
                f"{split}/loss/total": loss.detach().clone().mean(),
            }
            if self.disc_factor > 0:
                log_dict.update(
                    {
                        f"{split}/loss/p_rec": p_rec_loss.detach().mean(),
                        f"{split}/loss/g": g_loss.detach().mean(),
                        f"{split}/loss/g_weighted": g_weighted.detach().mean(),
                        f"{split}/scalars/r1_penalty": r1_penalty.detach(),
                    }
                )

            return loss, log_dict

        elif optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discr(inputs.contiguous().detach())
            logits_fake = self.discr(reconstructions.contiguous().detach())

            if (global_step < self.disc_start) or (self.training is False):
                d_loss = self.disc_factor * self.discr_loss(logits_real, logits_fake)
            else:
                d_loss = torch.tensor(0.0, requires_grad=True)

            log_dict = {
                f"{split}/loss/disc": d_loss.detach().clone().mean(),
                f"{split}/logits/real": logits_real.detach().mean(),
                f"{split}/logits/fake": logits_fake.detach().mean(),
            }
            return d_loss, log_dict

        else:
            raise ValueError(f"Unknown optimizer_idx {optimizer_idx}")
