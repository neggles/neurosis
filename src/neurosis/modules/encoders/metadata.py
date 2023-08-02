from typing import Tuple

from einops import rearrange
from torch import Tensor

from neurosis.modules.diffusion import Timestep
from neurosis.modules.diffusion.model import Encoder
from neurosis.modules.encoders.embedding import AbstractEmbModel
from neurosis.modules.regularizers import DiagonalGaussianRegularizer


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    def forward(self, x: Tensor):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb


class GaussianEncoder(Encoder, AbstractEmbModel):
    def __init__(self, weight: float = 1.0, flatten_output: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.posterior = DiagonalGaussianRegularizer()
        self.weight = weight
        self.flatten_output = flatten_output

    def forward(self, x) -> Tuple[dict[str, Tensor], Tensor]:
        z = super().forward(x)
        z, log = self.posterior(z)
        log["loss"] = log["kl_loss"]
        log["weight"] = self.weight
        if self.flatten_output:
            z = rearrange(z, "b c h w -> b (h w ) c")
        return log, z
