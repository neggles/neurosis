from typing import Any, Callable, Iterable, Optional, TypeAlias, Union

from torch import Tensor

Params: TypeAlias = Union[Iterable[Tensor], Iterable[dict[str, Any]]]
ParamGroup: TypeAlias = dict[str, Any]

LossClosure: TypeAlias = Callable[[], float]
OptLossClosure: TypeAlias = Optional[LossClosure]
Betas2: TypeAlias = tuple[float, float]
State: TypeAlias = dict[str, Any]
Nus2: TypeAlias = tuple[float, float]
