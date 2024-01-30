import logging
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch
from einops import rearrange
from matplotlib import (
    colormaps,
    pyplot as plt,
)
from torch import ParameterDict, Tensor, nn
from torch.nn import functional as F
from torchvision.utils import make_grid

from neurosis.modules.losses.functions import hinge_d_loss, vanilla_d_loss
from neurosis.modules.losses.lpips import LPIPS
from neurosis.modules.losses.patchgan import NLayerDiscriminator, weights_init

logger = logging.getLogger(__name__)


class GeneralLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int,
        logvar_init: float = 0.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: str = "hinge",
        scale_input_to_tgt_size: bool = False,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Union[None, dict[str, float]] = None,
        additional_log_keys: Optional[list[str]] = None,
        discriminator_config: Optional[dict] = None,
    ):
        super().__init__()
        self.dims = dims
        if self.dims > 2:
            logger.info(
                f"running with {dims=}. This means that for perceptual loss calculation, "
                + "the LPIPS loss will be applied to each frame independently. "
            )
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        if disc_loss not in ["hinge", "vanilla"]:
            raise ValueError(f"disc_loss must be one of ['hinge', 'vanilla'], got {disc_loss}")

        self.perceptual_loss = LPIPS()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar

        disc_kwargs = dict(input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False)
        if discriminator_config is not None:
            disc_kwargs.update(discriminator_config)

        self.discriminator = NLayerDiscriminator(**disc_kwargs).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.regularization_weights = regularization_weights or {}

        self.forward_keys = [
            "optimizer_idx",
            "global_step",
            "last_layer",
            "split",
            "regularization_log",
        ]

        self.additional_log_keys = set(additional_log_keys or [])
        self.additional_log_keys.update(set(self.regularization_weights.keys()))

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.discriminator.parameters()

    def get_trainable_autoencoder_parameters(self) -> Any:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    @torch.no_grad()
    def log_images(self, inputs: Tensor, reconstructions: Tensor) -> dict[str, Tensor]:
        # calc logits of real/fake
        logits_real = self.discriminator(inputs.contiguous().detach())
        if len(logits_real.shape) < 4:
            # Non patch-discriminator
            return dict()
        logits_fake = self.discriminator(reconstructions.contiguous().detach())
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

    def calculate_adaptive_weight(
        self,
        nll_loss: Tensor,
        g_loss: Tensor,
        last_layer: Tensor,
    ) -> Tensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def get_nll_loss(
        self,
        rec_loss: Tensor,
        weights: Optional[Union[float, Tensor]] = None,
    ) -> tuple[Tensor, Tensor]:
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        return nll_loss, weighted_nll_loss

    def forward(
        self,
        inputs: Tensor,
        reconstructions: Tensor,
        regularization_log: dict = {},
        optimizer_idx: int = 0,
        global_step: int = ...,
        last_layer: Optional[ParameterDict] = None,
        split: str = "train",
        weights: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict]:
        if self.scale_input_to_tgt_size:
            inputs = F.interpolate(inputs, reconstructions.shape[2:], mode="bicubic", antialias=True)

        if self.dims > 2:
            inputs, reconstructions = map(
                lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                (inputs, reconstructions),
            )

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss, weighted_nll_loss = self.get_nll_loss(rec_loss, weights)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if global_step >= self.discriminator_iter_start or not self.training:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
                if self.training:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            loss = weighted_nll_loss + d_weight * self.disc_factor * g_loss
            log = dict()
            for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                if k in self.additional_log_keys:
                    log[f"{split}/{k}"] = regularization_log[k].detach().float().mean()

            log.update(
                {
                    f"{split}/loss/total": loss.detach().clone().mean(),
                    f"{split}/loss/nll": nll_loss.detach().mean(),
                    f"{split}/loss/rec": rec_loss.detach().mean(),
                    f"{split}/loss/g": g_loss.detach().mean(),
                    f"{split}/scalars/logvar": self.logvar.detach(),
                    f"{split}/scalars/d_weight": d_weight.detach(),
                }
            )
            return loss, log

        elif optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            if global_step >= self.discriminator_iter_start or not self.training:
                d_loss = self.disc_factor * self.disc_loss(logits_real, logits_fake)
            else:
                d_loss = torch.tensor(0.0, requires_grad=True)

            log = {
                f"{split}/loss/disc": d_loss.detach().clone().mean(),
                f"{split}/logits/real": logits_real.detach().mean(),
                f"{split}/logits/fake": logits_fake.detach().mean(),
            }
            return d_loss, log

        else:
            raise ValueError(f"Unknown optimizer_idx {optimizer_idx}")
