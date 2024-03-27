from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from neurosis.models.diffusion import UNetModel, GeneralConditioner
from neurosis.modules.diffusion import Decoder, Encoder, ResnetBlock, MemoryEfficientAttnBlock
from neurosis.modules.diffusion.openaimodel import ResBlock, SpatialTransformer

# from neurosis.models.text_encoder import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from torch import nn


def diffusion_fsdp_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
) -> bool:
    return transformer_auto_wrap_policy(
        module,
        recurse,
        nonwrapped_numel,
        transformer_layer_cls={
            Encoder,
            Decoder,
            ResnetBlock,
            MemoryEfficientAttnBlock,
            GeneralConditioner,
            UNetModel,
            ResBlock,
            SpatialTransformer,
        },
    )
