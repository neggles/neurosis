# pytorch_diffusion + derived encoder decoder
import logging
import math
from typing import Any, Optional, Sequence
from warnings import warn

import numpy as np
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from neurosis.modules.attention import LinearAttention, MemoryEfficientCrossAttention
from neurosis.modules.layers import Normalize
from neurosis.modules.regularizers import DiagonalGaussianRegularizer

logger = logging.getLogger(__name__)

try:
    from xformers import ops as xops

    XFORMERS_IS_AVAILABLE = True
except ImportError:
    XFORMERS_IS_AVAILABLE = False

    xops = None
    logger.debug("xformers is not available, proceeding without it")


def get_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # asymmetric pad layer
            self.padding = nn.ConstantPad2d((0, 1, 0, 1), 0)
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        else:
            self.pooler = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        if self.with_conv:
            x = self.padding(x)
            x = self.conv(x)
        else:
            x = self.pooler(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def attention(self, h_: Tensor) -> Tensor:
        h_: Tensor = self.norm(h_)
        q: Tensor = self.q(h_)
        k: Tensor = self.k(h_)
        v: Tensor = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v))
        h_ = F.scaled_dot_product_attention(q, k, v)  # scale is dim ** -0.5 per default
        # compute attention

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.attention_op: Optional[Any] = None

    def attention(self, h_: Tensor) -> Tensor:
        h_: Tensor = self.norm(h_)
        q: Tensor = self.q(h_)
        k: Tensor = self.k(h_)
        v: Tensor = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = out.unsqueeze(0).reshape(B, 1, out.shape[1], C).permute(0, 2, 1, 3).reshape(B, out.shape[1], C)
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class TorchSDPAttnBlock(MemoryEfficientAttnBlock):
    def attention(self, h_: Tensor, mask: Tensor = None) -> Tensor:
        h_: Tensor = self.norm(h_)
        q: Tensor = self.q(h_)
        k: Tensor = self.k(h_)
        v: Tensor = self.v(h_)
        # single head
        B, C, H, W = q.shape
        heads = 1
        dim_head = C
        q, k, v = map(
            lambda t: t.view(B, -1, 1, C).transpose(1, 2),
            (q, k, v),
        )

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(B, -1, C)
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x: Tensor, context=None, mask=None, **kwargs) -> Tensor:
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        return x + out


def make_attn(in_channels: int, attn_type="vanilla", attn_kwargs=None) -> nn.Module:
    if attn_type not in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn",
        "torch-sdp",
        "linear",
        "none",
    ]:
        raise ValueError(f"attn_type {attn_type} unknown")

    logger.debug(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        if not XFORMERS_IS_AVAILABLE:
            raise ValueError("xformers is not available, cannot use vanilla-xformers attention")
        logger.debug(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "torch-sdp":
        return TorchSDPAttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        use_timestep: bool = True,
        use_linear_attn: bool = False,
        attn_type="vanilla",
    ):
        super().__init__()
        if use_linear_attn:
            warn("use_linear_attn is deprecated, use attn_type='linear' instead")
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList(
                [
                    nn.Linear(self.ch, self.temb_ch),
                    nn.Linear(self.temb_ch, self.temb_ch),
                ]
            )

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor, t: Optional[Tensor] = None, context: Optional[Tensor] = None) -> Tensor:
        # assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = F.silu(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Sequence[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        embed_dim: int = 256,
        standalone: bool = False,
        **kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            warn("use_linear_attn is deprecated, use attn_type='linear' instead")
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.regularizer = DiagonalGaussianRegularizer(sample=False)
        self.max_batch_size = None

        self.standalone = standalone
        self.quant_conv: nn.Conv2d | nn.Identity
        if self.standalone:
            quant_conv_in_ch = (1 + double_z) * z_channels
            quant_conv_out_ch = (1 + double_z) * embed_dim
            self.quant_conv = nn.Conv2d(quant_conv_in_ch, quant_conv_out_ch, 1)
        else:
            self.quant_conv = nn.Identity()

    def encode(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

    def forward(
        self,
        x: Tensor,
        regularize: bool = False,
    ):
        if self.max_batch_size is None:
            z = self.encode(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encode(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)
        if regularize:
            z, reg_log = self.regularizer(z)

        return z


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Sequence[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        tanh_out: bool = False,
        use_linear_attn: bool = False,
        attn_type: str = "vanilla",
        embed_dim: int = 256,
        standalone: bool = False,
        **kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            warn("use_linear_attn is deprecated, use attn_type='linear' instead")
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logger.debug(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)
        self.max_batch_size = None

        self.standalone = standalone
        self.post_quant_conv: nn.Conv2d | nn.Identity
        if self.standalone:
            self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        else:
            self.post_quant_conv = nn.Identity()

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

    def forward(
        self,
        z: Tensor,
        cat_zero: bool = False,
        **kwargs,
    ):
        if self.max_batch_size is None:
            z = self.post_quant_conv(z)
            dec = self.decode(z, **kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decode(dec_batch, **kwargs)
                dec.append(dec_batch)
            if cat_zero:
                dec = torch.cat(dec, 0)
        return dec
