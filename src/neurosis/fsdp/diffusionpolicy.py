from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from neurosis.modules.diffusion.model import Decoder, Encoder, ResnetBlock, TorchSDPAttnBlock
from neurosis.modules.diffusion.openaimodel import ResBlock, SpatialTransformer, UNetModel
from neurosis.modules.encoders import GeneralConditioner


class DiffusionFsdpPolicy(ModuleWrapPolicy):
    def __init__(self):
        module_classes = {
            Encoder,
            Decoder,
            ResnetBlock,
            TorchSDPAttnBlock,
            GeneralConditioner,
            UNetModel,
            ResBlock,
            SpatialTransformer,
        }

        super().__init__(module_classes=module_classes)
