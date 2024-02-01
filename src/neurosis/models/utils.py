import logging
from pathlib import Path

from diffusers import AutoencoderKL

from neurosis.constants import CHECKPOINT_EXTNS
from neurosis.modules.autoencoding.asymmetric import AsymmetricAutoencoderKL

logger = logging.getLogger(__name__)


def load_vae_ckpt(
    model_path: Path,
    asymmetric: bool = False,
    model_cls: type = None,
    **model_kwargs,
) -> AsymmetricAutoencoderKL | AutoencoderKL:
    model_cls = AsymmetricAutoencoderKL if asymmetric else AutoencoderKL

    if model_path.is_file():
        if model_path.suffix.lower() in CHECKPOINT_EXTNS:
            load_fn = model_cls.from_single_file
        else:
            raise ValueError(f"model file {model_path} is not a valid checkpoint file")
    elif model_path.is_dir():
        if model_path.joinpath("config.json").exists():
            load_fn = model_cls.from_pretrained
        else:
            raise ValueError(f"model folder {model_path} is not a HF checkpoint (no config.json)")
    else:
        raise ValueError(f"model path {model_path} is not a file or directory")

    return load_fn(model_path, **model_kwargs)
