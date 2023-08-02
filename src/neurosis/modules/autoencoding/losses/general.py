from typing import Any, Iterator, Optional

import torch
from einops import rearrange
from torch import ParameterDict, Tensor, nn
from torch.nn import Parameter

from neurosis.modules.losses.functions import adopt_weight, hinge_d_loss, vanilla_d_loss
from neurosis.modules.losses.lpips import LPIPS
from neurosis.modules.losses.patchgan import NLayerDiscriminator, weights_init


class GeneralLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int = ...,
        logvar_init: float = 0.0,
        pixelloss_weight: float = 1.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: str = "hinge",
        scale_input_to_tgt_size: bool = False,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Optional[dict] = None,
    ):
        super().__init__()
        self.dims = dims
        if self.dims > 2:
            print(
                f"running with dims={dims}. This means that for perceptual loss calculation, "
                f"the LPIPS loss will be applied to each frame independently. "
            )
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.regularization_weights = regularization_weights or {}

    def get_trainable_parameters(self) -> Iterator[Parameter]:
        return self.discriminator.parameters()

    def get_trainable_autoencoder_parameters(self) -> Any:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
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
        regularization_log: dict,
        inputs: Tensor,
        reconstructions: Tensor,
        optimizer_idx: int,
        global_step: int,
        last_layer: Optional[ParameterDict] = None,
        split: str = "train",
        weights: Optional[Tensor] = None,
    ):
        if self.scale_input_to_tgt_size:
            inputs = torch.nn.functional.interpolate(
                inputs, reconstructions.shape[2:], mode="bicubic", antialias=True
            )

        if self.dims > 2:
            inputs, reconstructions = map(
                lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                (inputs, reconstructions),
            )

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

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

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss
            log = dict()
            for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                log[f"{split}/{k}"] = regularization_log[k].detach().mean()

            log.update(
                {
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                }
            )

            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log
