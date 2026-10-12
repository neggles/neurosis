from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

from torch import Tensor

Params: TypeAlias = Iterable[Tensor] | Iterable[dict[str, Any]]
ParamGroup: TypeAlias = dict[str, Any]

LossClosure: TypeAlias = Callable[[], float]
OptLossClosure: TypeAlias = LossClosure | None
Betas2: TypeAlias = tuple[float, float]
State: TypeAlias = dict[str, Any]
Nus2: TypeAlias = tuple[float, float]
