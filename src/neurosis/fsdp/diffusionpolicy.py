from torch.distributed.fsdp.wrap import ModuleWrapPolicy

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


class DiffusionFsdpPolicy(ModuleWrapPolicy):
    def __init__(self):
        module_classes = {
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
        }

        super().__init__(module_classes=module_classes)
