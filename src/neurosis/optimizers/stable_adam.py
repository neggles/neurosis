from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

from .types import LossClosure, Params


class StableAdam(Optimizer):
    """
    StableAdamW PyTorch reimplementation.
    Reference: https://arxiv.org/pdf/2304.13013.pdf
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_thresh: float = 1.0,
        use_l2: bool = False,
        *,
        foreach: bool = False,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            use_l2=use_l2,
            foreach=foreach,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)
        self.eps = eps
        self.d = clip_thresh

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]["step"])
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    @_use_grad_for_differentiable
    def step(self, closure: LossClosure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            p: Parameter
            for p in group["params"]:
                if p.grad is None:
                    continue  # no gradient for this parameter, skip

                p_lr = group["lr"]
                p_beta1 = group["beta1"]
                p_beta2 = group["beta2"]
                p_decay = group["weight_decay"]

                # Cast gradients to float32 if necessary
                grad = p.grad.float() if p.dtype != torch.float32 else p.grad
                # get parameter state and gradient shape
                state = self.state[p]
                grad_shape = grad.shape

                if "step" not in state:
                    # initialize state for this parameter
                    state["grad_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["sqr_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["ones"] = torch.ones(1, dtype=p.dtype, device=p.device)
                    state["step"] = 0

                state["step"] += 1

        pass


def sa_step(
    p: Tensor,
    lr: float,
    eps: float,
    wd: float,
    beta1: float,
    beta2: float,
    step: int = 0,
    grad_avg: Optional[Tensor] = None,
    sqr_avg: Optional[Tensor] = None,
    use_l2: bool = True,
    do_wd: bool = True,
    eps_t=None,
    **kwargs,
):
    if step == 0:
        grad_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
        sqr_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
        eps_t = torch.tensor(eps, device=p.device, dtype=p.dtype)

    if wd != 0 and do_wd:
        if use_l2:
            # weight_decay
            p.data.mul_(1 - lr * wd)
        else:
            # expiramental l2_reg. not in paper
            p.grad.data.add_(p.data, alpha=wd)
    # calculate debiased momentum (beta) terms
    step += 1
    beta1hat = sa_debias(beta1, step)
    beta2hat = sa_debias(beta2, step)

    # update moving averages (average_grad & average_sqr_grad)
    grad_avg.mul_(beta1hat).add_(p.grad.data, alpha=1 - beta1hat)
    sqr_avg.mul_(beta2hat).addcmul_(p.grad.data, p.grad.data, value=1 - beta2hat)

    # compute per tensor RMS stabilization term
    root_sqr_avg = sqr_avg.sqrt()
    rms = torch.norm(p.grad.data.div(root_sqr_avg.maximum(eps_t)), 2)

    # calculate RMS stabilized η_t
    lr = sa_substep(rms, lr)

    # stable adam step
    p.data.addcdiv_(grad_avg, root_sqr_avg.add(eps_t), value=-lr)

    return {"grad_avg": grad_avg, "sqr_avg": sqr_avg, "step": step, "eps_t": eps_t}


def sa_foreach_step(
    p: list[Tensor],
    g: list[Tensor],
    grad_avg: list[Tensor],
    sqr_avg: list[Tensor],
    ones: list[Tensor],
    steps: np.ndarray[Any, int],
    do_wd: np.ndarray[Any, bool],
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
    use_l2: bool,
    **kwargs,
):
    if weight_decay != 0:
        if use_l2:
            wd = np.where(do_wd, 1 - lr * weight_decay, 1.0)
            torch._foreach_mul_(p, scalars=wd.tolist())
        else:
            # expiramental l2_reg. not in paper
            wd = np.where(do_wd, wd, 1.0)
            torch._foreach_addcdiv_(g, p, ones, scalars=wd.tolist())
            # cannot use scalers with foreach_add & multiple tensors, so divide by one with foreach_addcdiv

    # calculate debiased momentum (beta) terms
    beta1hat = (beta1**steps - beta1) / (beta1**steps - 1)
    beta2hat = (beta2**steps - beta2) / (beta2**steps - 1)

    # update moving average
    torch._foreach_mul_(grad_avg, scalars=beta1hat.tolist())
    torch._foreach_addcdiv_(grad_avg, g, ones, scalars=(1 - beta1hat).tolist())

    # update squared moving average
    torch._foreach_mul_(sqr_avg, scalars=beta2hat.tolist())
    torch._foreach_addcmul_(sqr_avg, g, g, scalars=(1 - beta2hat).tolist())

    # compute per tensor RMS stabilization term
    root_sqr_avg = torch._foreach_sqrt(sqr_avg)
    rms = torch._foreach_norm(torch._foreach_div(g, torch._foreach_maximum(root_sqr_avg, eps)), 2)

    # calculate RMS stabilized η_t
    lrs = [sa_substep(r, lr) for r in rms]

    torch._foreach_add_(root_sqr_avg, eps)
    torch._foreach_addcdiv_(p, grad_avg, root_sqr_avg, scalars=lrs)


## Functions below are split out to allow TorchScript compilation for performance


# optimized impl of:
# beta = beta*(1-beta**(step-1))/(1-beta**step)
@torch.jit.script
def sa_debias(beta: float, step: int):
    "StableAdam debiasing calculation"
    return (beta**step - beta) / (beta**step - 1)


@torch.jit.script
def sa_substep(rms: Tensor, lr: float):
    return -lr / max(1, rms.item())
