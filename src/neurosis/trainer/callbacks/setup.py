import logging
from os import PathLike
from pathlib import Path
from time import sleep
from typing import Optional, Union

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from omegaconf import OmegaConf

MULTINODE_HACKS = True

logger = logging.getLogger(__name__)


class TF32Callback(Callback):
    """Set up Torch's various tf32/matmul precision things"""

    def __init__(
        self,
        matmul: bool | str = "highest",
        cudnn: bool = False,
        cublas: bool = False,
    ):
        self.cudnn = cudnn
        self.cublas = cublas

        if isinstance(matmul, bool):
            if matmul is True:
                self.matmul = "high"
            elif matmul is False:
                self.matmul = "highest"
        else:
            if matmul not in ["medium", "high", "highest"]:
                raise ValueError(f"Invalid matmul value {matmul}")
            self.matmul = matmul

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.matmul != "highest":
            logger.info(f"Setting PyTorch matmul precision to '{self.matmul}'")
            torch.set_float32_matmul_precision(self.matmul)
        if hasattr(torch, "cuda") and torch.cuda.is_available() and self.cublas:
            logger.info("Enabling TensorFloat32 in cuBLAS library")
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available() and self.cudnn:
            logger.info("Enabling TensorFloat32 in cuDNN library")
            torch.backends.cudnn.allow_tf32 = True
        return super().setup(trainer, pl_module, stage)


class SetupCallback(Callback):
    def __init__(
        self,
        resume: bool = False,
        now: str = ...,
        logdir: PathLike = ...,
        ckptdir: PathLike = ...,
        cfgdir: PathLike = ...,
        config: Union[OmegaConf, dict] = ...,
        lightning_config: Union[OmegaConf, dict] = ...,
        debug: bool = False,
        ckpt_name: Optional[str] = None,
    ):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir: Path = Path(logdir)
        self.ckptdir: Path = Path(ckptdir)
        self.cfgdir: Path = Path(cfgdir)
        self.config = config if isinstance(config, OmegaConf) else OmegaConf.create(config)
        self.lightning_config = (
            lightning_config
            if isinstance(lightning_config, OmegaConf)
            else OmegaConf.create(lightning_config)
        )
        self.debug = debug
        self.ckpt_name = ckpt_name

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: Exception) -> None:
        if self.debug is False and trainer.is_global_zero:
            logger.info("Exception occurred - summoning checkpoint...")
            if self.ckpt_name is None:
                ckpt_path = self.ckptdir.joinpath("last.ckpt")
            else:
                ckpt_path = self.ckptdir.joinpath(self.ckpt_name)
            trainer.save_checkpoint(ckpt_path)
            logger.info("Checkpoint summoned.")

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        if trainer.is_global_zero:
            # Create logdirs and save configs
            self.logdir.mkdir(exist_ok=True, parents=True)
            self.ckptdir.mkdir(exist_ok=True, parents=True)
            self.cfgdir.mkdir(exist_ok=True, parents=True)

            if "callbacks" in self.lightning_config:
                if "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                    self.ckptdir.joinpath("trainstep_checkpoints").mkdir(exist_ok=True, parents=True)

            logger.info("Project config")
            logger.info(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                sleep(5)
            OmegaConf.save(
                self.config,
                self.cfgdir.joinpath(f"{self.now}-project.yaml"),
            )

            logger.info("Lightning config")
            logger.info(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                self.cfgdir.joinpath(f"{self.now}-lightning.yaml"),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not (MULTINODE_HACKS or self.resume) and self.logdir.exists():
                target_dir = self.logdir.parent.joinpath("child_runs", self.logdir.name)
                rank = trainer.local_rank
                logger.info(f"[R{rank}]: Renaming logdir {self.logdir} to {target_dir}")
                target_dir.mkdir(exist_ok=True, parents=True)
                try:
                    self.logdir.rename(target_dir.absolute())
                except FileNotFoundError:
                    pass
