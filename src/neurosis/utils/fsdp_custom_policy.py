from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from neurosis.models.diffusion import GeneralConditioner, UNetModel
from neurosis.modules.diffusion import Decoder, Encoder, ResnetBlock, TorchSDPAttnBlock
from neurosis.modules.diffusion.openaimodel import ResBlock, SpatialTransformer


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
            TorchSDPAttnBlock,
            GeneralConditioner,
            UNetModel,
            ResBlock,
            SpatialTransformer,
        },
    )