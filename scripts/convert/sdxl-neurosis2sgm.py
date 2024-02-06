#!/usr/bin/env python3
"""Convert PyTorch Lightning full-state checkpoint to a SafeTensors state dict."""
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from safetensors.torch import save_file
from typer import Typer

app = Typer(
    name="pl2sd",
    add_help_option=True,
    rich_help_panel=True,
)


def load_pl(path: PathLike) -> OrderedDict:
    path = Path(path).resolve()
    try:
        pl_sd = torch.load(str(path), map_location="cpu", mmap=True)
    except Exception:
        pl_sd = torch.load(str(path), map_location="cpu")

    return pl_sd["state_dict"]


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
            help="Path to the source PyTorch Lightning checkpoint file",
        ),
    ] = ...,
    out_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to save the SafeTensors checkpoint to (defaults to source with .safetensors suffix)",
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
        out_path = ckpt_path.with_suffix(".safetensors")

    out_path = Path(out_path).resolve()
    if out_path.is_dir():
        out_path = out_path.joinpath(ckpt_path.name).with_suffix(".safetensors")

    typer.echo("Loading PyTorch Lightning checkpoint...")
    pl_sd = load_pl(ckpt_path)

    typer.echo("Saving SafeTensors checkpoint...")
    save_safetensors(out_path, pl_sd)

    typer.echo("Done!")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
