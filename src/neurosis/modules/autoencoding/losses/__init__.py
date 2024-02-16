from typing import TypeAlias

from .discriminator_loss import GeneralLPIPSWithDiscriminator
from .dreamsim import AutoencoderDreamsim
from .latent_lpips import LatentLPIPS
from .vae_lpips_discr import AutoencoderLPIPSWithDiscr, AutoencoderPerceptual
from .vqperceptual import VQLPIPSWithDiscriminator

AutoencoderLoss: TypeAlias = (
    AutoencoderDreamsim
    | AutoencoderLPIPSWithDiscr
    | AutoencoderPerceptual
    | VQLPIPSWithDiscriminator
    | GeneralLPIPSWithDiscriminator
)

__all__ = [
    "AutoencoderDreamsim",
    "AutoencoderLPIPSWithDiscr",
    "AutoencoderPerceptual",
    "GeneralLPIPSWithDiscriminator",
    "LatentLPIPS",
    "VQLPIPSWithDiscriminator",
]
