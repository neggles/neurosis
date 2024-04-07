#!/usr/bin/env python3
"""Convert Stable Diffusion 1.5 full-state checkpoint to a Neurosis SafeTensors checkpoint."""

from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from safetensors.torch import load_file, save_file
from typer import Typer

app = Typer(
    name="pl2sd",
    add_help_option=True,
    rich_help_panel=True,
)


def rename_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    for k, v in state_dict.items():
        if "cond_stage_model." in k:
            k = k.replace("cond_stage_model.", "conditioner.embedders.0.", 1)
        new_state_dict[k] = v
    return new_state_dict


def load_ldm(path: PathLike) -> OrderedDict:
    path = Path(path).resolve()
    if path.suffix.lower() in [".ckpt", ".pt", ".pth"]:
        try:
            ldm_sd = torch.load(str(path), map_location="cpu", mmap=True)
        except Exception:
            ldm_sd = torch.load(str(path), map_location="cpu")
    elif path.suffix.lower() == ".safetensors":
        ldm_sd = load_file(path, device="cpu")
    else:
        raise ValueError(f"Unknown file extension {path.suffix}!")
    return ldm_sd


def save_safetensors(
    path: PathLike,
    state_dict: OrderedDict,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    path = Path(path).resolve()
    return save_file(state_dict, path, metadata)


@app.command()
def main(
    ckpt_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the source Stable Diffusion checkpoint file",
        ),
    ] = ...,
    out_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to save the SafeTensors checkpoint to (defaults to source with .neurosis.safetensors suffix)",
        ),
    ] = None,
):
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        ckpt_path = ckpt_path.with_suffix(".ckpt")
    if not ckpt_path.exists():
        ckpt_path = ckpt_path.with_suffix(".pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint file at {ckpt_path}!")
    if out_path is None:
        out_path = ckpt_path.with_suffix(".neurosis.safetensors")

    out_path = Path(out_path).resolve()
    if out_path.is_dir():
        out_path = out_path.joinpath(ckpt_path.name).with_suffix(".neurosis.safetensors")

    typer.echo(f"Loading Stable Diffusion checkpoint {ckpt_path.name}...")
    ldm_sd = load_ldm(ckpt_path)

    typer.echo("Remapping keys...")
    neurosis_sd = rename_keys(ldm_sd)

    typer.echo(f"Saving Neurosis checkpoint {out_path.name}...")
    save_safetensors(out_path, neurosis_sd)

    typer.echo("Done!")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
