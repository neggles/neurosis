import torch
from torch import Tensor, nn

from neurosis.modules.diffusion.openaimodel import UNetModel


class IdentityWrapper(nn.Module):
    def __init__(
        self,
        diffusion_model: UNetModel,
        compile_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.diffusion_model: UNetModel
        if compile_model:
            self.diffusion_model = torch.compile(diffusion_model, **kwargs)
        else:
            self.diffusion_model = diffusion_model

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        c: dict[str, Tensor],
        **kwargs,
    ) -> Tensor:
        x = torch.cat((x, c.get("concat", Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )
