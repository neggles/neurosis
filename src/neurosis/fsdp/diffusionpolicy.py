from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, Type, Union

import lightning.pytorch as pl
import torch
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies.fsdp import FSDPStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer

from neurosis.models.autoencoder import AutoencoderKL, FSDPAutoencoderKL
from neurosis.models.autoencoder_hf import AutoencoderKL as HFAutoencoderKL
from neurosis.models.text_encoder import (
    FrozenCLIPEmbedder,
    FrozenOpenCLIPEmbedder2,
    FrozenT5Embedder,
)
from neurosis.modules.diffusion.model import Decoder, Encoder
from neurosis.modules.diffusion.openaimodel import (
    SpatialTransformer,
    TimestepEmbedSequential,
    UNetModel,
)
from neurosis.modules.encoders import GeneralConditioner
from neurosis.utils.misc import str_to_dtype

if TYPE_CHECKING:
    from torch.distributed.fsdp import CPUOffload, ShardingStrategy

    _POLICY = Union[set[Type[Module]], Callable[[Module, bool, int], bool], ModuleWrapPolicy]
    _SHARDING_STRATEGY = Union[
        ShardingStrategy,
        Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"],
    ]


class DiffusionFsdpPolicy(ModuleWrapPolicy):
    def __init__(self):
        module_classes = {
            CLIPAttention,
            CLIPEncoderLayer,
            Decoder,
            Encoder,
            FrozenCLIPEmbedder,
            FrozenOpenCLIPEmbedder2,
            FrozenT5Embedder,
            GeneralConditioner,
            SpatialTransformer,
            TimestepEmbedSequential,
            UNetModel,
        }

        super().__init__(module_classes=module_classes)


@dataclass
class SDXLMixedPrecision:
    param_dtype: Optional[str | torch.dtype] = None
    reduce_dtype: Optional[str | torch.dtype] = None
    buffer_dtype: Optional[str | torch.dtype] = None
    keep_low_precision_grads: bool = False
    cast_forward_inputs: bool = False
    cast_root_forward_inputs: bool = True
    _module_classes_to_ignore: Sequence[Type[torch.nn.Module]] = (_BatchNorm,)
    tenc_fp32: bool = False
    vae_fp32: bool = False

    def __post_init__(self):
        if isinstance(self.param_dtype, str):
            self.param_dtype = str_to_dtype(self.param_dtype)
        if isinstance(self.reduce_dtype, str):
            self.reduce_dtype = str_to_dtype(self.reduce_dtype)
        if isinstance(self.buffer_dtype, str):
            self.buffer_dtype = str_to_dtype(self.buffer_dtype)

        fp32_classes = [_BatchNorm]
        if self.tenc_fp32:
            fp32_classes.extend(
                [FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2, FrozenT5Embedder, GeneralConditioner],
            )

        if self.vae_fp32:
            fp32_classes.extend(
                [AutoencoderKL, Decoder, Encoder, FSDPAutoencoderKL, HFAutoencoderKL],
            )

        fp32_classes = sorted(set(fp32_classes), key=lambda x: x.__name__)
        self._module_classes_to_ignore = fp32_classes

    def as_torch_dataclass(self):
        return MixedPrecision(
            param_dtype=self.param_dtype,
            reduce_dtype=self.reduce_dtype,
            buffer_dtype=self.buffer_dtype,
            keep_low_precision_grads=self.keep_low_precision_grads,
            cast_forward_inputs=self.cast_forward_inputs,
            cast_root_forward_inputs=self.cast_root_forward_inputs,
            _module_classes_to_ignore=self._module_classes_to_ignore,
        )


class SDXLFSDPStrategy(FSDPStrategy):
    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[list[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[Precision] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        cpu_offload: Union[bool, "CPUOffload", None] = None,
        mixed_precision: Optional[SDXLMixedPrecision] = None,
        auto_wrap_policy: Optional["_POLICY"] = None,
        activation_checkpointing: Optional[Union[Type[Module], list[Type[Module]]]] = None,
        activation_checkpointing_policy: Optional["_POLICY"] = None,
        sharding_strategy: "_SHARDING_STRATEGY" = "FULL_SHARD",
        state_dict_type: Literal["full", "sharded"] = "full",
        **kwargs: Any,
    ) -> None:
        if mixed_precision is not None:
            mixed_precision = mixed_precision.as_torch_dataclass()

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
            timeout=timeout,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_policy=activation_checkpointing_policy,
            sharding_strategy=sharding_strategy,
            state_dict_type=state_dict_type,
            **kwargs,
        )
