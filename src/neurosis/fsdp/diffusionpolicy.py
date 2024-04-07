from dataclasses import InitVar
from typing import Optional

import torch
from clip.model import ResidualAttentionBlock
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

from neurosis.models.autoencoder import AutoencoderKL, AutoencoderKLInferenceWrapper
from neurosis.models.autoencoder_hf import AutoencoderKL as HFAutoencoderKL
from neurosis.models.text_encoder import (
    FrozenCLIPEmbedder,
    FrozenOpenCLIPEmbedder2,
    FrozenT5Embedder,
)
from neurosis.modules.diffusion.model import (
    Decoder,
    Encoder,
)
from neurosis.modules.diffusion.openaimodel import (
    SpatialTransformer,
    TimestepEmbedSequential,
    UNetModel,
)
from neurosis.modules.encoders.embedding import GeneralConditioner


class DiffusionFsdpPolicy(ModuleWrapPolicy):
    def __init__(self):
        module_classes = {
            CLIPEncoderLayer,
            Decoder,
            Encoder,
            FrozenCLIPEmbedder,
            FrozenOpenCLIPEmbedder2,
            FrozenT5Embedder,
            GeneralConditioner,
            ResidualAttentionBlock,
            SpatialTransformer,
            TimestepEmbedSequential,
            UNetModel,
        }

        super().__init__(module_classes=module_classes)


def get_mixed_precision(
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
    buffer_dtype: Optional[torch.dtype] = None,
    keep_low_precision_grads: bool = False,
    cast_forward_inputs: bool = False,
    cast_root_forward_inputs: bool = True,
    tenc_fp32: bool = False,
    vae_fp32: bool = False,
):
    fp32_classes = []
    if tenc_fp32:
        fp32_classes.extend([FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2, FrozenT5Embedder])

    if vae_fp32:
        fp32_classes.extend([Decoder, Encoder, AutoencoderKL, AutoencoderKLInferenceWrapper, HFAutoencoderKL])

    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        keep_low_precision_grads=keep_low_precision_grads,
        cast_forward_inputs=cast_forward_inputs,
        cast_root_forward_inputs=cast_root_forward_inputs,
        _module_classes_to_ignore=fp32_classes if len(fp32_classes) > 0 else None,
    )


class SDXLMixedPrecision(MixedPrecision):
    tenc_fp32: InitVar[bool] = False
    vae_fp32: InitVar[bool] = False

    def __post_init__(self, tenc_fp32: bool, vae_fp32: bool):
        fp32_classes = []
        if tenc_fp32:
            fp32_classes.extend(
                [FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2, FrozenT5Embedder, GeneralConditioner],
            )

        if vae_fp32:
            fp32_classes.extend(
                [AutoencoderKL, Decoder, Encoder, HFAutoencoderKL],
            )

        fp32_classes = sorted(set(fp32_classes), key=lambda x: x.__name__)
        if len(fp32_classes) > 0:
            self._module_classes_to_ignore = fp32_classes
