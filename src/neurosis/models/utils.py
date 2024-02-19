import logging
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from neurosis.constants import CHECKPOINT_EXTNS

logger = logging.getLogger(__name__)


def load_vae_ckpt(
    model_path: Path,
    asymmetric: bool = False,  # not currently implemented
    **model_kwargs,
) -> AutoencoderKL:
    if asymmetric is not False:
        raise NotImplementedError("asymmetric VAE is currently not implemented")

    if model_path.is_file():
        if model_path.suffix.lower() in CHECKPOINT_EXTNS:
            load_fn = AutoencoderKL.from_single_file
        else:
            raise ValueError(f"model file {model_path} is not a valid checkpoint file")
    elif model_path.is_dir():
        if model_path.joinpath("config.json").exists():
            load_fn = AutoencoderKL.from_pretrained
        else:
            raise ValueError(f"model folder {model_path} is not a HF checkpoint (no config.json)")
    else:
        raise ValueError(f"model path {model_path} is not a file or directory")

    return load_fn(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=False, **model_kwargs)
