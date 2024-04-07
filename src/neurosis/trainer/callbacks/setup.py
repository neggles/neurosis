import logging

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

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
