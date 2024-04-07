"""
Functions in this file are courtesty of @ashen-sensored on GitHub - thankyou so much! <3

Used to merge DreamSim LoRA weights into the base ViT models manually, so we don't need
to use an ancient version of PeFT that is no longer supported (and kind of broken)
"""

import logging
from os import PathLike
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from .model import DreamsimModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def calculate_merged_weight(
    lora_a: Tensor,
    lora_b: Tensor,
    base: Tensor,
    scale: float,
    qkv_switches: list[bool],
) -> Tensor:
    n_switches = len(qkv_switches)
    n_groups = sum(qkv_switches)

    qkv_mask = torch.tensor(qkv_switches, dtype=torch.bool).reshape(len(qkv_switches), -1)
    qkv_mask = qkv_mask.broadcast_to((-1, base.shape[0] // n_switches)).reshape(-1)

    lora_b = lora_b.squeeze()
    delta_w = base.new_zeros(lora_b.shape[0], base.shape[1])

    grp_in_ch = lora_a.shape[0] // n_groups
    grp_out_ch = lora_b.shape[0] // n_groups
    for i in range(n_groups):
        islice = slice(i * grp_in_ch, (i + 1) * grp_in_ch)
        oslice = slice(i * grp_out_ch, (i + 1) * grp_out_ch)
        delta_w[oslice, :] = lora_b[oslice, :] @ lora_a[islice, :]

    delta_w_full = base.new_zeros(base.shape)
    delta_w_full[qkv_mask, :] = delta_w

    merged = base + scale * delta_w_full
    return merged.to(base)


@torch.no_grad()
def merge_dreamsim_lora(
    base_model: nn.Module,
    lora_path: PathLike,
    torch_device: torch.device | str = torch.device("cpu"),
):
    lora_path = Path(lora_path)
    # make sure model is on device
    base_model = base_model.eval().requires_grad_(False).to(torch_device)

    # load the lora
    if lora_path.suffix.lower() in [".pt", ".pth", ".bin"]:
        lora_sd = torch.load(lora_path, map_location=torch_device, weights_only=True)
    elif lora_path.suffix.lower() == ".safetensors":
        lora_sd = load_file(lora_path)
    else:
        raise ValueError(f"Unsupported file extension '{lora_path.suffix}'")

    # these loras were created by a cursed PEFT version, okay? so we have to do some crimes.
    group_prefix = "base_model.model.base_model.model.model."
    # get all lora weights for qkv layers, stripping the insane prefix
    group_weights = {k.replace(group_prefix, ""): v for k, v in lora_sd.items() if k.startswith(group_prefix)}
    # strip ".lora_X.weight" from keys to match against base model keys
    group_layers = set([k.rsplit(".", 2)[0] for k in group_weights.keys()])

    base_weights = base_model.state_dict()
    for key in [x for x in base_weights.keys() if "attn.qkv.weight" in x]:
        param_name = key.rsplit(".", 1)[0]
        if param_name not in group_layers:
            logger.warning(f"QKV param '{param_name}' not found in lora weights")
            continue
        new_weight = calculate_merged_weight(
            group_weights[f"{param_name}.lora_A.weight"],
            group_weights[f"{param_name}.lora_B.weight"],
            base_weights[key],
            0.5 / 16,
            [True, False, True],
        )
        base_weights[key] = new_weight

    base_model.load_state_dict(base_weights)
    return base_model.requires_grad_(False)


def remap_clip(state_dict: dict[str, Tensor], variant: str) -> dict[str, Tensor]:
    """Remap keys from the original DreamSim checkpoint to match new model structure."""

    def prepend_extractor(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        if variant.endswith("single"):
            return {f"extractor.{k}": v for k, v in state_dict.items()}
        return state_dict

    if "clip" not in variant:
        return prepend_extractor(state_dict)

    if "patch_embed.proj.bias" in state_dict:
        _ = state_dict.pop("patch_embed.proj.bias", None)
    if "pos_drop.weight" in state_dict:
        state_dict["norm_pre.weight"] = state_dict.pop("pos_drop.weight")
        state_dict["norm_pre.bias"] = state_dict.pop("pos_drop.bias")
    if "head.weight" in state_dict and "head.bias" not in state_dict:
        state_dict["head.bias"] = torch.zeros(state_dict["head.weight"].shape[0])

    return prepend_extractor(state_dict)


def convert_dreamsim_single(
    ckpt_path: PathLike,
    variant: str,
    ensemble: bool = False,
) -> DreamsimModel:
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists():
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path.joinpath("ensemble" if ensemble else variant)
            ckpt_path = ckpt_path.joinpath(f"{variant}_merged.safetensors")

    # defaults are for dino, overridden as needed below
    patch_size = 16
    layer_norm_eps = 1e-6
    pre_norm = False
    act_layer = "gelu"

    match variant:
        case "open_clip_vitb16" | "open_clip_vitb32" | "clip_vitb16" | "clip_vitb32":
            patch_size = 32 if "b32" in variant else 16
            layer_norm_eps = 1e-5
            pre_norm = True
            img_mean = (0.48145466, 0.4578275, 0.40821073)
            img_std = (0.26862954, 0.26130258, 0.27577711)
            act_layer = "quick_gelu" if variant.startswith("clip_") else "gelu"
        case "dino_vitb16":
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
        case _:
            raise NotImplementedError(f"Unsupported model variant '{variant}'")

    model: DreamsimModel = DreamsimModel(
        image_size=224,
        patch_size=patch_size,
        layer_norm_eps=layer_norm_eps,
        pre_norm=pre_norm,
        act_layer=act_layer,
        img_mean=img_mean,
        img_std=img_std,
    )
    state_dict = load_file(ckpt_path, device="cpu")
    state_dict = remap_clip(state_dict)
    model.extractor.load_state_dict(state_dict)
    return model
