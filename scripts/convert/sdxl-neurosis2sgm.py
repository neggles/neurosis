#!/usr/bin/env python3
"""Convert PyTorch Lightning full-state checkpoint to a SafeTensors state dict."""
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from safetensors.torch import save_file
from tqdm import tqdm
from typer import Typer

try:
    from rich.pretty import print
    from rich.traceback import install as install_traceback

    _ = install_traceback(show_locals=True, locals_max_length=0)
except ImportError:
    pass

app = Typer(
    name="pl2sd",
    add_help_option=True,
    rich_help_panel=True,
)


def load_pl(path: PathLike) -> tuple[OrderedDict, dict]:
    path = Path(path).resolve()
    try:
        pl_sd = torch.load(str(path), map_location="cpu", mmap=True)
    except Exception:
        print(f"Failed to load {path} with mmap=True, retrying with mmap=False")
        pl_sd = torch.load(str(path), map_location="cpu")

    metadata = {
        k: v
        for k, v in pl_sd.items()
        if k in ["epoch", "global_step", "pytorch-lightning_version", "hyper_parameters"]
    }
    metadata["ckpt_name"] = path.name
    return pl_sd["state_dict"], metadata


def save_safetensors(
    path: PathLike,
    state_dict: OrderedDict,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    path = Path(path).resolve()
    try:
        save_file(state_dict, path, metadata)
    except Exception as e:
        print(f"Failed to save SafeTensors checkpoint to {path}, trying without metadata")
        save_file(state_dict, path)
        raise RuntimeError("Failed to save SafeTensors checkpoint metadata!") from e

    print(f"SafeTensors checkpoint saved to {path}")


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
    fp16: Annotated[
        bool,
        typer.Option(
            "--fp16",
            is_flag=True,
            help="Cast weights to FP16 before saving the SafeTensors checkpoint",
        ),
    ] = False,
    fp32: Annotated[
        bool,
        typer.Option(
            "--fp32",
            is_flag=True,
            help="Cast weights to FP32 before saving the SafeTensors checkpoint",
        ),
    ] = False,
    with_metadata: Annotated[
        bool,
        typer.Option(
            "--metadata",
            is_flag=True,
            help="Include metadata in the SafeTensors checkpoint",
        ),
    ] = False,
):
    if fp16 and fp32:
        raise ValueError("Cannot cast to both FP16 and FP32 at the same time!")

    # resolve the input checkpoint path
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        ckpt_path = ckpt_path.with_suffix(".ckpt")
    if not ckpt_path.exists():
        ckpt_path = ckpt_path.with_suffix(".pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint file at {ckpt_path}!")

    # if output path is a directory, use the input filename
    if out_path is None:
        out_path = ckpt_path.with_suffix(".safetensors")

    # resolve the output path
    out_path = Path(out_path).resolve()
    if out_path.is_dir():
        out_path = out_path.joinpath(ckpt_path.name).with_suffix(".safetensors")

    # load the original checkpoint
    typer.echo("Loading PyTorch Lightning checkpoint...")
    state_dict, metadata = load_pl(ckpt_path)

    # cast weights to desired dtype if requested
    if fp16 or fp32:
        torch_dtype = torch.float16 if fp16 else torch.float16
        for k in tqdm(state_dict, desc=f"Casting weights to {torch_dtype}", unit="param"):
            if isinstance(state_dict[k], torch.Tensor):
                state_dict[k] = state_dict[k].to(torch_dtype)

    # save the SafeTensors checkpoint
    typer.echo("Saving SafeTensors checkpoint...")
    save_safetensors(out_path, state_dict, metadata if with_metadata else None)

    typer.echo("Done!")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
