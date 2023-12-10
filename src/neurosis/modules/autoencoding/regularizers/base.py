from abc import abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: Tensor) -> Tuple[Tensor, dict]:
        raise NotImplementedError("Abstract base class was called ;_;")

    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError("Abstract base class was called ;_;")


class IdentityRegularizer(AbstractRegularizer):
    def forward(self, z: Tensor) -> Tuple[Tensor, dict]:
        return z, dict()

    def get_trainable_parameters(self) -> Any:
        yield from ()


def measure_perplexity(predicted_indices: Tensor, num_centroids: int) -> Tuple[Tensor, Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use
