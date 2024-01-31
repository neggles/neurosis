import logging
from typing import Any, Generator, Iterator, Optional

import torch
from torch import Tensor, nn
from torch.nn import Parameter

from neurosis.modules.losses.functions import apply_threshold_weight, get_discr_loss_fn
from neurosis.modules.losses.patchgan import NLayerDiscriminator
from neurosis.modules.losses.perceptual import LPIPS
from neurosis.modules.losses.types import DiscriminatorLoss

logger = logging.getLogger(__name__)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int = ...,
        logvar_init: float = 0.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        codebook_weight: float = 1.0,
        pixelloss_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: DiscriminatorLoss = DiscriminatorLoss.Hinge,
        learn_logvar: bool = False,
    ):
        super().__init__()
        self.disc_loss = get_discr_loss_fn(disc_loss)

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=False,
        )
        self.discriminator.initialize_weights()
        self.discriminator_iter_start = disc_start

        logger.info(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def get_trainable_parameters(self) -> Iterator[Parameter]:
        return self.discriminator.parameters()

    def get_trainable_autoencoder_parameters(self) -> Generator[Parameter, Any, None]:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None) -> Tensor:
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        codebook_loss: Tensor,
        inputs: Tensor,
        reconstructions: Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: Optional[Tensor] = None,
        split: str = "train",
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = apply_threshold_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            loss = (
                nll_loss + (d_weight * disc_factor * g_loss) + (self.codebook_weight * codebook_loss.mean())
            )

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/p_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log

        elif optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = apply_threshold_weight(
                self.disc_factor, global_step, start_step=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log

        else:
            raise ValueError("optimizer_idx must be 0 or 1")
