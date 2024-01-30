import logging
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Optional

import torch
from dinov2 import DinoVisionTransformer
from dinov2.models.vision_transformer import vit_base, vit_giant2, vit_large, vit_small
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class DinoV2Config(str, Enum):
    Small = "dinov2_vits14"
    Base = "dinov2_vitb14"
    Large = "dinov2_vitl14"
    Giant = "dinov2_vitg14"
    SmallRegistered = "dinov2_vits14_reg"
    BaseRegistered = "dinov2_vitb14_reg"
    LargeRegistered = "dinov2_vitl14_reg"
    GiantRegistered = "dinov2_vitg14_reg"


def hub_load_dinov2(
    variant: DinoV2Config,
    trust_repo: bool = True,
    **kwargs,
):
    """
    Loads a DINOv2 model from the official repository
    """
    model_name: str = DinoV2Config(variant).value

    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2",
        model=model_name,
        source="github",
        trust_repo=trust_repo,
        **kwargs,
    )
    return model


def create_dinov2(config: DinoV2Config, ckpt_path: Optional[PathLike] = None) -> DinoVisionTransformer:
    """
    Creates a DINOv2 model from a config and optionally loads weights from a checkpoint
    """
    model: DinoVisionTransformer
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)

    common_kwargs = dict(
        img_size=518,
        patch_size=14,
        block_chunks=0,
        init_values=1.0,
    )
    base_kwargs = dict(
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    reg_kwargs = dict(
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    )

    match config:
        case DinoV2Config.Small:
            model = vit_small(ffn_layer="mlp", **common_kwargs, **base_kwargs)
        case DinoV2Config.Base:
            model = vit_base(ffn_layer="mlp", **common_kwargs, **base_kwargs)
        case DinoV2Config.Large:
            model = vit_large(ffn_layer="mlp", **common_kwargs, **base_kwargs)
        case DinoV2Config.Giant:
            model = vit_giant2(ffn_layer="swiglufused", **common_kwargs, **base_kwargs)
        case DinoV2Config.SmallRegistered:
            model = vit_small(ffn_layer="mlp", **common_kwargs, **reg_kwargs)
        case DinoV2Config.BaseRegistered:
            model = vit_base(ffn_layer="mlp", **common_kwargs, **reg_kwargs)
        case DinoV2Config.LargeRegistered:
            model = vit_large(ffn_layer="mlp", **common_kwargs, **reg_kwargs)
        case DinoV2Config.GiantRegistered:
            model = vit_giant2(ffn_layer="swiglufused", **common_kwargs, **reg_kwargs)
        case _:
            raise ValueError(f"Invalid DinoV2Config: {config}")

    if ckpt_path is not None:
        logger.info(f"Loading weights from {ckpt_path}")
        if ckpt_path.suffix.lower() == ".safetensors":
            state_dict = load_file(ckpt_path)
        elif ckpt_path.suffix.lower() in [".pt", ".pth", ".bin"]:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        else:
            raise ValueError(f"Invalid checkpoint file: {ckpt_path}")
        _ = model.load_state_dict(state_dict)
    else:
        logger.info("No checkpoint path provided, looking for default...")
        state_dict = None
        for fpath in Path("data/dinov2").iterdir():
            if fpath.stem == config:
                logger.info(f"Found weights at {fpath}, loading...")
                if fpath.suffix.lower() == ".safetensors":
                    state_dict = load_file(fpath)
                elif fpath.suffix.lower() in [".pt", ".pth", ".bin"]:
                    state_dict = torch.load(fpath, map_location="cpu")
                _ = model.load_state_dict(state_dict)
                break
        if state_dict is None:
            logger.warning("No weights found, using default initialization")

    return model
