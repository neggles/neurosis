from torch import Tensor, nn


class FreezeSliceHook(nn.Module):
    def __init__(self, slices: tuple[slice, ...]):
        super().__init__()
        self.slices = slices

    def forward(self, param: Tensor) -> None:
        param[self.slices].grad = None
        pass
