from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, Optional

import torch
import typer
from safetensors.torch import load_file, save_file
from torch import Tensor

app = typer.Typer()


class AdjustmentOp(str, Enum):
    add = "add"
    multiply = "mul"
    divide = "div"
    subtract = "sub"


def decoder_op_to_fn(op: AdjustmentOp) -> Callable[[Tensor, Tensor], Tensor]:
    match op:
        case AdjustmentOp.add:
            return torch.add
        case AdjustmentOp.multiply:
            return torch.mul
        case AdjustmentOp.divide:
            return torch.div
        case AdjustmentOp.subtract:
            return torch.sub
        case _:
            raise ValueError(f"Unknown operator: {op}")


def encoder_op_to_fn(op: AdjustmentOp) -> Callable[[Tensor, Tensor], Tensor]:
    match op:
        case AdjustmentOp.add:
            return torch.sub
        case AdjustmentOp.multiply:
            return torch.div
        case AdjustmentOp.divide:
            return torch.mul
        case AdjustmentOp.subtract:
            return torch.add
        case _:
            raise ValueError(f"Unknown operator: {op}")


def load_state_dict(model_path: Path) -> dict[str, Tensor]:
    model_path = Path(model_path).resolve()
    match model_path.suffix.lower():
        case ".pt" | ".pth" | ".bin":
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        case ".safetensors":
            state_dict = load_file(model_path, device="cpu")
        case _:
            raise ValueError(f"Unsupported file extension '{model_path.suffix}'")
    return state_dict


def save_state_dict(state_dict: dict[str, Tensor], output_path: Path):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match output_path.suffix.lower():
        case ".pt" | ".pth" | ".bin":
            torch.save(state_dict, output_path)
        case ".safetensors":
            save_file(state_dict, output_path)
        case _:
            raise ValueError(f"Unsupported file extension '{output_path.suffix}'")


ADJUST_KEYS = {
    "decoder": {
        "contrast": "decoder.conv_out.weight",
        "brightness": "decoder.conv_out.bias",
    },
    "encoder": {
        "contrast": "encoder.conv_in.weight",
        "brightness": "encoder.conv_in.bias",
    },
}


@app.command(no_args_is_help=True)
def main(
    model_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the model file in LDM format (.safetensors, .pt, .pth, .bin, etc.)",
        ),
    ],
    output_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to save the blessed model; defaults to <source_path>.blessed.safetensors",
        ),
    ] = None,
    contrast: Annotated[
        float,
        typer.Option(
            "--contrast",
            "-c",
            help="Contrast adjustment factor. 1.0 leaves the contrast unchanged.",
        ),
    ] = 1.0,
    contrast_op: Annotated[
        AdjustmentOp,
        typer.Option(
            "--contrast-op",
            "-C",
            help="Operator for contrast adjustment, defaults to multiply.",
        ),
    ] = AdjustmentOp.multiply,
    brightness: Annotated[
        float,
        typer.Option(
            "--brightness",
            "-b",
            help="Brightness adjustment factor. 0.0 leaves the brightness unchanged.",
        ),
    ] = 0.0,
    brightness_op: Annotated[
        AdjustmentOp,
        typer.Option(
            "--brightness-op",
            "-B",
            help="Operator for brightness adjustment, defaults to add",
        ),
    ] = AdjustmentOp.add,
    patch_encoder: Annotated[
        bool,
        typer.Option(
            "--patch-encoder",
            "-P",
            help="Apply the adjustment to encoder and decoder (default is decoder-only)",
            is_flag=True,
        ),
    ] = False,
):
    """'Bless' a VAE by adjusting the brightness and contrast in the final layer."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # set up output path
    if output_path is None:
        output_path = model_path.with_suffix(".blessed.safetensors")

    typer.echo(f"Loading model from '{model_path}'...")
    state_dict = load_state_dict(model_path)

    # cast  the whole thing to float32 in case it wasnt already
    typer.echo("Ensuring model is in fp32 for adjustments")
    state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}

    for key in ADJUST_KEYS["decoder"].values():
        if key.endswith("weight"):
            typer.echo(f"Adjusting decoder contrast by {contrast:.2f} using {contrast_op}")
            state_dict[key] = decoder_op_to_fn(contrast_op)(state_dict[key], contrast)
        elif key.endswith("bias"):
            typer.echo(f"Adjusting decoder brightness by {brightness:.2f} using {brightness_op}")
            state_dict[key] = decoder_op_to_fn(brightness_op)(state_dict[key], brightness)

    if patch_encoder:
        for key in ADJUST_KEYS["encoder"].values():
            if key.endswith("weight"):
                typer.echo(f"Adjusting encoder contrast by {contrast:.2f} using {contrast_op}")
                state_dict[key] = encoder_op_to_fn(contrast_op)(state_dict[key], contrast)
            elif key.endswith("bias"):
                typer.echo(f"Adjusting encoder brightness by {brightness:.2f} using {brightness_op}")
                state_dict[key] = encoder_op_to_fn(brightness_op)(state_dict[key], brightness)
    else:
        typer.echo("Skipping encoder adjustments")

    typer.echo(f"Saving blessed model to '{output_path}'...")
    save_state_dict(state_dict, output_path)

    typer.echo("Done! enjoy :V")


if __name__ == "__main__":
    app()
