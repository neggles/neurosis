import torch
from torch import Tensor, nn


class Normalize(nn.GroupNorm):
    def __init__(self, in_channels: int, num_groups: int = 32):
        super().__init__(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ActNorm(nn.Module):
    initialized: Tensor

    def __init__(
        self,
        num_features: int,
        logdet: bool = False,
        affine: bool = True,
        allow_reverse_init: bool = False,
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input: Tensor):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input: Tensor, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output: Tensor) -> Tensor:
        if self.training and self.initialized.item() == 0:
            if self.allow_reverse_init:
                self.initialize(output)
                self.initialized.fill_(1)
            else:
                raise RuntimeError(
                    "ActNorm has not been initialized with forward pass, "
                    + "use .initialize() first or set allow_reverse_init=True"
                )

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h
