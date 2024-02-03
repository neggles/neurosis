from typing import TypeAlias

from .discriminator_loss import GeneralLPIPSWithDiscriminator
from .lpips import LatentLPIPS
from .vae_lpips_discr import AutoencoderLPIPSWithDiscr, AutoencoderPerceptual
from .vqperceptual import VQLPIPSWithDiscriminator

AutoencoderLoss: TypeAlias = (
    AutoencoderLPIPSWithDiscr
    | AutoencoderPerceptual
    | VQLPIPSWithDiscriminator
    | GeneralLPIPSWithDiscriminator
)

__all__ = [
    "AutoencoderLPIPSWithDiscr",
    "AutoencoderPerceptual",
    "GeneralLPIPSWithDiscriminator",
    "LatentLPIPS",
    "VQLPIPSWithDiscriminator",
]
