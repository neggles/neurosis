#!/usr/bin/env python3
"""Convert neurosis (Lightning) VAE checkpoint into HF model and ldm checkpoint"""

from collections import OrderedDict
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Optional

import torch
import typer
from diffusers import AutoencoderKL
from safetensors.torch import save_file
from typer import Typer

try:
    from rich.traceback import install as traceback_install

    _ = traceback_install(show_locals=False, width=120)
except ImportError:
    pass

app = Typer(
    name="pl2sd-vae",
    add_help_option=True,
    rich_help_panel=True,
)


class AutoencoderType(str, Enum):
    SDXL = "sdxl"
    SD15 = "sd1.5"


VAE_CONFIGS = {
    AutoencoderType.SDXL: {
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        "force_upcast": False,
        "in_channels": 3,
        "latent_channels": 4,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 512,
        "scaling_factor": 0.13025,
        "up_block_types": [
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
    },
}


def load_pl(path: PathLike) -> OrderedDict:
    path = Path(path).resolve()
    try:
        pl_sd = torch.load(str(path), map_location="cpu", mmap=True)
    except Exception:
        pl_sd = torch.load(str(path), map_location="cpu")

    pl_sd = pl_sd["state_dict"]
    vae_sd = {k.replace("vae.", ""): v for k, v in pl_sd.items() if k.startswith("vae.")}
    return vae_sd


def save_safetensors(
    path: PathLike,
    state_dict: OrderedDict,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    path = Path(path).resolve()
    save_file(state_dict, path, metadata)


def save_huggingface(
    path: PathLike,
    state_dict: OrderedDict,
    config: dict[str, Any],
    push: bool = False,
    repo_id: Optional[str] = None,
    push_kwargs: dict[str, Any] = {},
) -> None:
    path = Path(path).resolve()
    vae = AutoencoderKL.from_config(config)
    vae.load_state_dict(state_dict)
    vae = vae.eval().requires_grad_(False)
    vae.save_pretrained(path, safe_serialization=True)
    if push is True:
        if repo_id is None:
            raise ValueError("repo_id is required to push to the Hub")
        vae.push_to_hub(repo_id, safe_serialization=True, **push_kwargs)


def resolve_ckpt_path(path: PathLike) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        for extn in [".ckpt", ".pt", ".pth", ".safetensors"]:
            if path.with_suffix(extn).is_file():
                return path.with_suffix(extn)
        raise FileNotFoundError(f"Could not find checkpoint file at '{path}'!")
    else:
        return path


def resolve_out_path(
    ckpt_path: Path,
    out_path: Optional[Path],
    diffusers: bool = False,
    no_ckpt: bool = False,
) -> tuple[Optional[Path], Optional[Path]]:
    if no_ckpt is True:
        diffusers = True

    if out_path is None:
        out_path = ckpt_path
    out_name = out_path.stem

    if diffusers is True:
        if out_path.exists():
            out_path = out_path.parent.joinpath(out_name)
        if out_path.exists() and out_path.is_file():
            out_path = out_path.parent.joinpath(f"{out_name}-diffusers")
        elif out_path.exists() and out_path.is_dir():
            if len(list(out_path.iterdir())) == 0:
                # empty dir, remove it
                out_path.unlink()
            else:
                out_path = out_path.joinpath(f"{out_name}-diffusers")
        hf_path = out_path
        hf_path.mkdir(parents=True)
        out_path = out_path.joinpath(out_name + ".safetensors")

    else:
        hf_path = None
        out_path = Path(out_path).resolve().with_suffix(".safetensors")
        if out_path.is_file():
            out_path = out_path.parent.joinpath(out_name + "-ldm.safetensors")
        if out_path.is_file():
            raise FileExistsError(f"Output file already exists at {out_path}!")
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if no_ckpt is True:
        out_path = None

    return out_path, hf_path


@app.command()
def main(
    ckpt_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the source neurosis/Lightning checkpoint file",
            dir_okay=False,
            exists=True,
            readable=True,
        ),
    ] = ...,
    out_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to save the converted model to, if directory, will use the source name. Must be directory for HF output",
            writable=True,
        ),
    ] = None,
    diffusers: Annotated[
        bool,
        typer.Option(
            "--diffusers",
            "-D",
            help="Save as HuggingFace Diffusers model",
            flag_value=True,
        ),
    ] = False,
    hf_config: Annotated[
        AutoencoderType,
        typer.Option(
            "--model-config",
            "-C",
            help="Model configuration to use for Diffusers conversion (not needed for LDM)",
            case_sensitive=False,
        ),
    ] = AutoencoderType.SDXL,
    no_ckpt: Annotated[
        bool,
        typer.Option(
            "--no-ldm",
            "-N",
            help="Do not save the SafeTensors LDM checkpoint (implies --diffusers)",
            flag_value=True,
        ),
    ] = False,
    push_to_hub: Annotated[
        bool,
        typer.Option(
            "--push-to-hub",
            "-P",
            help="Push the converted model to the HuggingFace Hub",
            flag_value=True,
        ),
    ] = False,
    hf_repo_id: Annotated[
        Optional[str],
        typer.Option(
            "--repo",
            "-R",
            help="HuggingFace Hub repository ID to push to (implies --push-to-hub, required if --push-to-hub)",
        ),
    ] = None,
):
    if push_to_hub:
        if hf_repo_id is None:
            raise ValueError("Cannot push to Hub without a repo ID!")
        typer.echo("Push to hub requested, enabling Diffusers output")
        diffusers = True

    # resolve input file
    ckpt_path = resolve_ckpt_path(ckpt_path)
    typer.echo(f"Found neurosis checkpoint: {ckpt_path}")

    out_path, hf_path = resolve_out_path(ckpt_path, out_path, diffusers, no_ckpt)
    if hf_path is not None:
        typer.echo(f"HuggingFace Diffusers output dir: {hf_path}")
    if out_path is not None:
        typer.echo(f"SafeTensors checkpoint output path: {out_path}")

    typer.echo("Loading neurosis checkpoint...")
    pl_sd = load_pl(ckpt_path)

    if hf_path is not None:
        typer.echo("Saving HuggingFace Diffusers model...")
        save_huggingface(hf_path, pl_sd, VAE_CONFIGS[hf_config], push_to_hub, hf_repo_id)

    if out_path is not None:
        typer.echo("Saving SafeTensors checkpoint...")
        save_safetensors(out_path, pl_sd)

    typer.echo("Done!")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
