import logging
import warnings
from typing import Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalVAEMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution, Encoder
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from torch import Tensor, nn

from neurosis.modules.autoencoding.asym_decoder import Decoder

logger = logging.getLogger(__name__)


class AsymmetricAutoencoderKL(ModelMixin, ConfigMixin, FromOriginalVAEMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        down_block_out_channels: tuple[int, ...] = (64,),
        layers_per_down_block: int = 1,
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        up_block_out_channels: tuple[int, ...] = (64,),
        layers_per_up_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups_encoder: int = 32,
        norm_num_groups_decoder: int = 64,
        attn_scale_sqrt_inv: Optional[int] = None,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        block_out_channels=None,
        force_upcast: bool = False,
    ) -> None:
        super().__init__()
        attn_scale = None
        if attn_scale_sqrt_inv is not None:
            attn_scale = 1.0 / attn_scale_sqrt_inv**0.5

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=down_block_out_channels,
            layers_per_block=layers_per_down_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups_encoder,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=up_block_out_channels,
            layers_per_block=layers_per_up_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups_decoder,
            attn_scale=attn_scale,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        self.register_to_config(block_out_channels=up_block_out_channels)
        self.register_to_config(force_upcast=False)

    @apply_forward_hook
    def encode(self, x: Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple[Tensor]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: Tensor, return_dict: bool = True) -> DecoderOutput | tuple[Tensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: Tensor, return_dict: bool = True, generator: Optional[torch.Generator] = None
    ) -> DecoderOutput | tuple[Tensor]:
        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> DecoderOutput | tuple[Tensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def enable_tiling(self):
        """No-op method, this model does not support tiling."""
        warnings.warn("This model does not support tiling.", UserWarning)

    def disable_tiling(self):
        """No-op method, this model does not support tiling."""
        pass

    def enable_slicing(self):
        """No-op method, this model does not support slicing."""
        warnings.warn("This model does not support slicing.", UserWarning)

    def disable_slicing(self):
        """No-op method, this model does not support slicing."""
        pass
