from typing import Any, Callable, Iterable, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class HybridOptimizer(Optimizer):
    """
    Wrapper around multiple optimizers that should be stepped together at a single time. This is
    a hack to avoid PyTorch Lightning calling ``training_step`` once for each optimizer, which
    increases training time and is not always necessary.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687

    Parameters
    ----------
    optimizers: list of optimizers

    """

    def __init__(self, optimizers: Iterable[Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def state(self) -> dict[str, Tensor]:
        """Return the combined state for each optimizer in ``self.optimizers``."""
        return {key: value for optimizer in self.optimizers for key, value in optimizer.state.items()}

    @property
    def param_groups(self) -> list[dict[str, Union[Tensor, float, bool, Any]]]:
        """Return the combined parameter groups for each optimizer in ``self.optimizers``."""
        return [element for optimizer in self.optimizers for element in optimizer.param_groups]

    @property
    def defaults(self) -> dict[str, Tensor]:
        """Return the combined defaults for each optimizer in ``self.optimizers``."""
        return {key: value for optimizer in self.optimizers for key, value in optimizer.defaults.items()}

    def __getstate__(self) -> list[Optimizer]:
        """Return ``self.optimizers`` for pickling purposes."""
        return self.optimizers

    def __setstate__(self, optimizers: list[Optimizer]) -> None:
        """
        Load ``optimizers`` into ``self.optimizers`` for pickling purposes and call
        ``__setstate__``.

        """
        self.optimizers = optimizers

        # call remaining lines of the ``Optimizer.__setstate__`` method just to be safe.
        # copied from: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()  # To support multiprocessing pickle/unpickle.
            optimizer.defaults.setdefault("differentiable", False)

    def __repr__(self) -> str:
        """Call and concatenate ``__repr__`` for each optimizer in ``self.optimizers``."""
        repr_str = f"``{self.__class__.__name__}`` containing {len(self.optimizers)} optimizers:\n"

        for optimizer in self.optimizers:
            repr_str += "\n" + optimizer.__repr__()

        return repr_str

    def _hook_for_profile(self) -> None:
        """Call ``_hook_for_profile`` for each optimizer in ``self.optimizers``."""
        for optimizer in self.optimizers:
            optimizer._hook_for_profile()

    def state_dict(
        self,
    ) -> list[dict[str, Union[Tensor, list[dict[str, Union[Tensor, float, bool, Any]]]]]]:
        """
        Returns the state of the optimizer as a dictionary.

        It contains two entries:

            * ``state`` - a dict holding current optimization state. Its content differs between
              optimizer classes.
            * ``param_groups`` - a list containing all parameter groups where each parameter group
              is a dict

        """
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(
        self,
        state_dict: list[dict[str, Union[Tensor, list[dict[str, Union[Tensor, float, bool, Any]]]]]],
    ) -> None:
        """
        Loads the optimizer state.

        Parameters
        ----------
        state_dict: dict
            Optimizer state. Should be an object returned from a call to ``state_dict()``

        """
        for state, optimizer in zip(state_dict, self.optimizers):
            optimizer.load_state_dict(state)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Sets the gradients of all optimized ``Tensor``s to zero.

        Parameters
        ----------
        set_to_none: bool
            Instead of setting to zero, set the grads to ``None``. This will in general have lower
            memory footprint, and can modestly improve performance. However, it changes certain
            behaviors. For example:

                1. When the user tries to access a gradient and perform manual ops on it, a ``None``
                   attribute or a ``Tensor`` full of ``0``s will behave differently.

                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass,
                   ``.grad``s are guaranteed to be ``None`` for params that did not receive a
                   gradient.

                3. ``torch.optim`` optimizers have a different behavior if the gradient is ``0`` or
                   ``None`` (in one case it does the step with a gradient of ``0`` and in the other
                   it skips the step altogether).

        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], Tensor] = None) -> Tensor:
        """
        Performs a single optimization step (parameter update).

        Parameters
        ----------
        closure: function
            A closure that reevaluates the model and returns the loss. Optional for most optimizers.

        Notes
        -----
        Unless otherwise specified, this function should not modify the ``.grad`` field of the
        parameters.

        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for optimizer in self.optimizers:
            optimizer.step()

        return loss


class HybridScheduler(LRScheduler):
    """
    Wrapper class around ``lr_scheduler``s to return a dummy optimizer to pass PyTorch Lightning
    checks.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687

    Parameters
    ----------
    hybrid_optimizer: HybridOptim
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    idx: int
        Index of the optimizer in ``hybrid_optimizer`` the learning rate scheduler ``lr_scheduler``
        is assigned to

    """

    def __init__(
        self,
        hybrid_optimizer: HybridOptimizer,
        lr_scheduler: LRScheduler,
        optimizer_idx: int,
    ) -> None:
        self.optimizer = hybrid_optimizer
        self.lr_scheduler = lr_scheduler
        self.idx = optimizer_idx

    def __getattribute__(self, __name: str) -> Any:
        """
        If the attribute name is one of ``optimizer``, ``idx``, or ``lr_scheduler``, return this
        class's attribute with the same name, else return the ``lr_scheduler``'s attribute with
        that name.

        """
        if __name in {"optimizer", "lr_scheduler", "idx"}:
            return super().__getattribute__(__name)
        else:
            return self.lr_scheduler.__getattribute__(__name)
