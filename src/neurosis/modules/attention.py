import logging
import math
from functools import wraps
from typing import Any, Iterable, Optional

import torch
from einops import rearrange, repeat
from packaging import version
from torch import FloatTensor, Tensor, nn
from torch.backends.cuda import SDPBackend, sdp_kernel
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

try:
    from xformers import (
        __version__ as xformers_version,
        ops as xops,
    )

    XFORMERS_IS_AVAILABLE = True
    XFORMERS_VER = version.parse(xformers_version)
except ImportError:
    XFORMERS_IS_AVAILABLE = False
    logger.warn("xformers is not available, proceeding without it")

BACKEND_MAP = {
    SDPBackend.MATH: {
        "enable_math": True,
        "enable_flash": False,
        "enable_mem_efficient": False,
    },
    SDPBackend.FLASH_ATTENTION: {
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": False,
    },
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False,
        "enable_flash": False,
        "enable_mem_efficient": True,
    },
    None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
}


def uniq(arr: Iterable[Any]) -> list:
    return list(set(arr))


def max_neg_value(t: Tensor) -> FloatTensor:
    return -torch.finfo(t.dtype).max


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@wraps(nn.GroupNorm.__init__)
def get_norm_layer(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SelfAttention(nn.Module):
    ATTENTION_MODES = ("xformers", "torch", "math")

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_mode: str = "xformers",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.shape

        qkv = self.qkv(x)
        if self.attn_mode == "torch":
            qkv = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = F.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "B H L D -> B L (H D)")
        elif self.attn_mode == "xformers":
            qkv = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xops.memory_efficient_attention(q, k, v)
            x = rearrange(x, "B L H D -> B L (H D)", H=self.num_heads)
        elif self.attn_mode == "math":
            qkv = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplementedError(f"Unknown attention mode: {self.attn_mode}")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = get_norm_layer(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_: Tensor = x
        h_ = self.norm(h_)
        q: Tensor = self.q(h_)
        k: Tensor = self.k(h_)
        v: Tensor = self.v(h_)

        # compute attention
        _, C, H, _ = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k)

        w_ = w_ * (int(C) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=H)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        backend: Optional[SDPBackend] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout) if dropout else nn.Identity(),
        )
        self.backend = backend

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        additional_tokens: Optional[Tensor] = None,
        n_times_crossframe_attn_in_self: int = 0,
    ) -> Tensor:
        h = self.heads

        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)

        q = self.to_q(x)
        context = context or x
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self
            k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)
            v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # logger.debug("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        logger.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout) if dropout else nn.Identity(),
        )
        self.attention_op: Optional[Any] = None

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        additional_tokens: Optional[Tensor] = None,
        n_times_crossframe_attn_in_self: int = 0,
    ):
        if additional_tokens is not None:
            # get the number of masked tokens at the beginning of the output sequence
            n_tokens_to_mask = additional_tokens.shape[1]
            # add additional token
            x = torch.cat([additional_tokens, x], dim=1)
        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        if n_times_crossframe_attn_in_self:
            # reprogramming cross-frame attention as in https://arxiv.org/abs/2303.13439
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # n_cp = x.shape[0]//n_times_crossframe_attn_in_self
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        if XFORMERS_VER >= version.parse("0.0.21"):
            # TODO: Remove this after facebookresearch/xformers/issues/845 is fixed
            max_bs = 32768
            N = q.shape[0]
            n_batches = math.ceil(N / max_bs)
            out = list()
            for i_batch in range(n_batches):
                batch = slice(i_batch * max_bs, (i_batch + 1) * max_bs)
                out.append(
                    xops.memory_efficient_attention(
                        q[batch], k[batch], v[batch], attn_bias=None, op=self.attention_op
                    )
                )
            out = torch.cat(out, 0)
        else:
            out = xops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # TODO: Use this directly in the attention operation, as a bias
        if mask is not None:
            raise NotImplementedError("Masking is not implemented yet")
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if additional_tokens is not None:
            # remove additional token
            out = out[:, n_tokens_to_mask:]
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,  # ampere
    }

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: int = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        disable_self_attn: bool = False,
        attn_mode: str = "softmax",
        sdp_backend: Optional[SDPBackend] = None,
    ):
        super().__init__()
        if attn_mode not in self.ATTENTION_MODES:
            raise ValueError(f"Unknown attention mode: {attn_mode}")

        attn_cls = self.ATTENTION_MODES[attn_mode]
        if not isinstance(sdp_backend, (SDPBackend, type(None))):
            raise ValueError(f"Invalid SDP backend: {sdp_backend}")

        self.disable_self_attn = disable_self_attn
        # attn1 is a self-attention if not self.disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            backend=sdp_backend,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # is self-attn if context is none
        self.attn2 = attn_cls(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=sdp_backend,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        if self.checkpoint:
            logger.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        additional_tokens: Optional[Tensor] = None,
        n_times_crossframe_attn_in_self: int = 0,
    ) -> Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context, additional_tokens, n_times_crossframe_attn_in_self)

    def _forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        additional_tokens: Optional[Tensor] = None,
        n_times_crossframe_attn_in_self: int = 0,
    ) -> Tensor:
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
                if not self.disable_self_attn
                else 0,
            )
            + x
        )
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerSingleLayerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention  # on the A100s not quite as fast as the above version
        # (todo might depend on head_dim, check, falls back to semi-optimized kernels for dim!=[16,32,64,128])
    }

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: int = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        attn_mode: str = "softmax",
    ):
        super().__init__()
        if attn_mode not in self.ATTENTION_MODES:
            raise ValueError(f"Unknown attention mode: {attn_mode}")

        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.ff(self.norm2(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int | list[int]] = None,
        disable_self_attn: bool = False,
        use_linear: bool = False,
        attn_type: str = "softmax",
        use_checkpoint: bool = True,
        sdp_backend: Optional[SDPBackend] = None,
    ):
        super().__init__()
        logger.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads"
        )

        if context_dim is not None and not isinstance(context_dim, (list)):
            context_dim = [context_dim]
        if context_dim is not None and isinstance(context_dim, list):
            if depth != len(context_dim):
                logger.debug(
                    f"{self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth

        self.in_channels = in_channels
        self.norm = get_norm_layer(in_channels)

        inner_dim = n_heads * d_head
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                BasicTransformerBlock(
                    dim,
                    heads,
                    dim_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attn_mode="softmax-xformers",
                    checkpoint=checkpoint,
                )
            )

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, context)
        return x
