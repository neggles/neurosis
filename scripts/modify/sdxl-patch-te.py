#!/usr/bin/env python3
"""Replace the text encoders in an SDXL checkpoint with the ones from another checkpoint."""

from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from safetensors.torch import load_file, save_file
from torch import Tensor
from typer import Typer

try:
    from rich.pretty import print
    from rich.traceback import install as traceback_install

    _ = traceback_install(show_locals=False, width=120)
except ImportError:
    pass

app = Typer(
    name="sdxl-patch-te",
    add_help_option=True,
    rich_help_panel=True,
)


def load_safetensors(path: PathLike) -> OrderedDict | dict:
    path = Path(path).resolve()
    print(f"Loading SafeTensors checkpoint from {path}")
    state_dict = load_file(path)
    return state_dict


def save_safetensors(
    path: PathLike,
    state_dict: OrderedDict,
) -> None:
    path = Path(path).resolve()
    try:
        save_file(state_dict, path)
    except Exception as e:
        raise RuntimeError(f"Failed to save SafeTensors checkpoint to {path}!") from e
    print(f"SafeTensors checkpoint saved to {path}")


def replace_tenc(
    base_state_dict: OrderedDict | dict[str, Tensor],
    tenc_state_dict: OrderedDict | dict[str, Tensor],
    tenc_key_prefix: str = "conditioner.",
) -> OrderedDict[str, Tensor]:
    new_sd = base_state_dict.copy()
    print("Replaced keys:")
    for k in tenc_state_dict.keys():
        if k.startswith(tenc_key_prefix):
            typer.echo(f" - {k}")
            new_sd[k] = tenc_state_dict[k]
    print("Replacement complete!")
    return new_sd


@app.command()
def main(
    base_ckpt_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the SafeTensors checkpoint with the Unet/VAE you want to keep",
        ),
    ] = ...,
    tenc_ckpt_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the SafeTensors checkpoint with the text encoders you want to inject",
        ),
    ] = ...,
    out_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to save the SafeTensors checkpoint to (defaults to source with '_te' suffix)",
        ),
    ] = None,
):
    if base_ckpt_path.suffix.lower() != ".safetensors":
        raise ValueError("Base checkpoint must be a SafeTensors checkpoint!")
    if tenc_ckpt_path.suffix.lower() != ".safetensors":
        raise ValueError("TE checkpoint must be a SafeTensors checkpoint!")

    # resolve the input checkpoint path
    base_ckpt_path = Path(base_ckpt_path).resolve()
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Could not find base checkpoint file at {base_ckpt_path}!")
    tenc_ckpt_path = Path(tenc_ckpt_path).resolve()
    if not tenc_ckpt_path.exists():
        raise FileNotFoundError(f"Could not find TE checkpoint file at {tenc_ckpt_path}!")

    # resolve the output path
    if out_path is None:
        out_path = base_ckpt_path.parent.joinpath(base_ckpt_path.stem + "_te.safetensors")
    out_path = Path(out_path).resolve()
    if out_path.is_dir():
        out_path = out_path.parent.joinpath(base_ckpt_path.stem + "_te.safetensors")

    # load the original checkpoint
    typer.echo("Loading base checkpoint...")
    base_state_dict = load_safetensors(base_ckpt_path)
    typer.echo("Loading TE checkpoint...")
    tenc_state_dict = load_safetensors(tenc_ckpt_path)

    # merge the TE checkpoint's TE keys into the base checkpoint
    typer.echo("Replacing base checkpoint text encoder weights with TE donor weights...")
    merged_state_dict = replace_tenc(base_state_dict, tenc_state_dict)

    # save the SafeTensors checkpoint
    typer.echo("Saving merged SafeTensors checkpoint...")
    save_safetensors(out_path, merged_state_dict)

    typer.echo("Done!")
    raise typer.Exit(0)


if __name__ == "__main__":
    # disable gradients and set default device to CPU
    _ = torch.set_grad_enabled(False)
    torch.set_default_device("cpu")
    app()
