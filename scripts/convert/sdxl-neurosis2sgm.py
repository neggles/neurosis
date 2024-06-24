#!/usr/bin/env python3
"""Convert PyTorch Lightning full-state checkpoint to a SafeTensors state dict."""

import gc
import json
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm
from typer import Typer

try:
    from rich.pretty import print
    from rich.traceback import install as traceback_install

    _ = traceback_install(show_locals=False, width=120)
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
        k: json.dumps(v, default=str, ensure_ascii=False)
        for k, v in pl_sd.items()
        if k in ["epoch", "global_step", "pytorch-lightning_version", "hyper_parameters"]
    }
    metadata["ckpt_name"] = path.name
    return pl_sd["state_dict"], metadata


def save_safetensors(
    path: PathLike,
    state_dict: OrderedDict,
    metadata: Optional[dict[str, str]] = None,
    temp_dir: Optional[PathLike] = None,
) -> None:
    path = Path(path).resolve()
    save_path = Path(temp_dir).resolve().joinpath(path.name) if temp_dir else path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        save_file(state_dict, save_path, metadata)
    except Exception as e:
        print(f"Failed to save SafeTensors checkpoint with metadata to {path}: {e}")
        print(f"Metadata was: {metadata}")
        print("Trying without metadata...")
        save_file(state_dict, save_path)
        raise RuntimeError("Failed to save SafeTensors checkpoint metadata!") from e

    if temp_dir:
        path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("rb") as f:
            with path.open("wb") as out_f:
                out_f.write(f.read())
        save_path.unlink()

    print(f"SafeTensors checkpoint saved to {path}")


def maybe_remap_keys(
    state_dict: OrderedDict | dict[str, Tensor],
) -> OrderedDict[str, Tensor]:
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        # deal with moving the quant_conv and post_quant_conv back
        if k.startswith("vae_decoder.post_quant_conv."):
            # vae_decoder.post_quant_conv -> first_stage_model.post_quant_conv
            k = k.replace("vae_decoder.", "first_stage_model.")
        if k.startswith("vae_encoder.quant_conv."):
            # vae_encoder.quant_conv -> first_stage_model.quant_conv
            k = k.replace("vae_encoder.", "first_stage_model.")

        # generally remap the vae keys to first_stage_model
        if k.startswith("vae_"):
            # vae_encoder -> first_stage_model.encoder, etc
            k = k.replace("vae_", "first_stage_model.")
        # assign to new state dict
        new_sd[k] = v
    return new_sd


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
    temp_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--temp-dir",
            help="Temporary directory to use for saving the SafeTensors checkpoint before moving to the final location",
        ),
    ] = None,
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
        for idx, k in enumerate(tqdm(state_dict, desc=f"Casting weights to {torch_dtype}", unit="param")):
            if isinstance(state_dict[k], torch.Tensor):
                state_dict[k] = state_dict[k].to(torch_dtype)
            if idx % 100 == 0:
                gc.collect()
        gc.collect()

    # maybe remap vae keys
    state_dict = maybe_remap_keys(state_dict)

    # save the SafeTensors checkpoint
    typer.echo("Saving SafeTensors checkpoint...")
    save_safetensors(out_path, state_dict, metadata if with_metadata else None, temp_dir=temp_dir)

    typer.echo("Done!")
    raise typer.Exit(0)


if __name__ == "__main__":
    # disable gradients and set default device to CPU
    _ = torch.set_grad_enabled(False)
    torch.set_default_device("cpu")
    app()
