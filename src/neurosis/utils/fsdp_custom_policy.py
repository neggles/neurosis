from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from neurosis.models.embedding import GeneralConditioner
from neurosis.models.text_encoder import (
    FrozenCLIPEmbedder,
    FrozenOpenCLIPEmbedder,
    FrozenOpenCLIPEmbedder2,
    FrozenT5Embedder,
)
from neurosis.modules.diffusion.model import (
    AttnBlock,
    Decoder,
    Encoder,
    LinAttnBlock,
    MemoryEfficientAttnBlock,
    MemoryEfficientCrossAttentionWrapper,
    ResnetBlock,
    TorchSDPAttnBlock,
)
from neurosis.modules.diffusion.openaimodel import ResBlock, SpatialTransformer, UNetModel


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
            # AutoencodingEngine,
            AttnBlock,
            Decoder,
            Encoder,
            FrozenCLIPEmbedder,
            FrozenOpenCLIPEmbedder,
            FrozenOpenCLIPEmbedder2,
            FrozenT5Embedder,
            GeneralConditioner,
            LinAttnBlock,
            MemoryEfficientAttnBlock,
            MemoryEfficientCrossAttentionWrapper,
            ResBlock,
            ResnetBlock,
            SpatialTransformer,
            TorchSDPAttnBlock,
            UNetModel,
        },
    )
