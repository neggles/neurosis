import logging
from copy import deepcopy
from typing import Generator, Optional

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class LitEma(nn.Module):
    decay: Tensor
    num_updates: Tensor

    def __init__(self, model: nn.Module, decay=0.9999, use_num_updates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer("num_updates", torch.tensor(0 if use_num_updates else -1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace(".", "_")
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    def update(self, model: nn.Module):
        self.forward(model)

    def forward(self, model: nn.Module):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                elif key in self.m_name2s_name:
                    raise ValueError(f"Parameter {key} is not trainable, but has a shadow parameter")

    def copy_to(self, model: nn.Module):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            elif key in self.m_name2s_name:
                raise ValueError(f"Parameter {key} is not trainable, but has a shadow parameter")

    def store(self, parameters: list[Tensor]):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters: list[Tensor]):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 2/3.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    init_done: Tensor
    step: Tensor

    def __init__(
        self,
        model: nn.Module,
        ema_model: Optional[nn.Module] = None,  # if None, will be deepcopied from model
        beta: float = 0.9999,
        karras_beta: bool = False,  # if True, uses the karras time dependent beta
        update_after_step: int = 100,
        update_every: int = 10,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        no_ema_names: list[str] = [],
        ignore_names: list[str] = [],
        ignore_startswith: list[str] = [],
        include_online_model: bool = True,  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
        auto_move_device: bool = False,  # if the EMA model is on a different device (say CPU), automatically move the tensor
    ):
        super().__init__()
        self._beta = beta
        self.karras_beta = karras_beta
        self.is_frozen = beta == 1.0

        no_ema_names = set(no_ema_names)
        ignore_names = set(ignore_names)
        ignore_startswith = set(ignore_startswith)

        # whether to include the online model within the module tree, so that state_dict also saves it
        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]  # hack

        # ema model
        if ema_model is None:
            try:
                self.ema_model: nn.Module = deepcopy(model)
            except Exception as e:
                logger.exception(f"Unable to deepcopy model {model.__class__.__name__}: {e}")
                raise RuntimeError("Unable to deepcopy model (lazy tensors?)") from e
        else:
            self.ema_model: nn.Module = ema_model

        # disable gradients for the EMA model
        self.ema_model.requires_grad_(False)

        # parameter and buffer names
        self.parameter_names = {
            name
            for name, param in self.ema_model.named_parameters()
            if param.dtype in [torch.float32, torch.float16, torch.bfloat16]
        }
        self.buffer_names = {
            name
            for name, buffer in self.ema_model.named_buffers()
            if buffer.dtype in [torch.float32, torch.float16, torch.bfloat16]
        }

        # updating hyperparameters
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        if not isinstance(no_ema_names, (set, list)):
            raise ValueError("no_ema_names must be a set or list")

        self.no_ema_names = no_ema_names  # parameter or buffer
        self.ignore_names = ignore_names
        self.ignore_startswith = ignore_startswith

        # whether to manage if EMA model is kept on a different device
        self.auto_move_device = auto_move_device

        # init and step states
        self.register_buffer("init_done", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0))

    def inplace_copy(self, src: Tensor, tgt: Tensor):
        if self.auto_move_device:
            tgt = tgt.to(src.device)
        src.copy_(tgt)

    def inplace_lerp(self, src: Tensor, tgt: Tensor, weight: float):
        if self.auto_move_device:
            tgt = tgt.to(src.device)
        src.lerp_(tgt, weight)

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    @property
    def beta(self):
        if self.karras_beta:
            return (1 - 1 / (self.step + 1)) ** (1 + self.power)
        return self._beta

    @property
    def current_decay(self):
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.0)
        if epoch <= 0:
            return 0.0

        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return value.clamp(min=self.min_value, max=self.beta).item()

    def eval(self) -> "EMA":
        self.ema_model.eval()
        return self

    def restore_ema_model_device(self):
        device = self.init_done.device
        self.ema_model.to(device)

    def get_params(self, model: nn.Module) -> Generator[tuple[str, nn.Parameter], None, None]:
        yield from ((name, param) for name, param in model.named_parameters() if name in self.buffer_names)

    def get_buffers(self, model: nn.Module) -> Generator[tuple[str, Tensor], None, None]:
        yield from ((name, buffer) for name, buffer in model.named_buffers() if name in self.buffer_names)

    def copy_model_to_ema(self):
        ema_param: nn.Parameter
        model_param: nn.Parameter
        ema_buffer: Tensor
        model_buffer: Tensor

        for (_, ema_param), (_, model_param) in zip(
            self.get_params(self.ema_model), self.get_params(self.model)
        ):
            self.inplace_copy(ema_param.data, model_param.data)

        for (_, ema_buffer), (_, model_buffer) in zip(
            self.get_buffers(self.ema_model), self.get_buffers(self.model)
        ):
            self.inplace_copy(ema_buffer.data, model_buffer.data)

    def copy_ema_to_model(self):
        ema_param: nn.Parameter
        model_param: nn.Parameter
        ema_buffer: Tensor
        model_buffer: Tensor

        for (_, ema_param), (_, model_param) in zip(
            self.get_params(self.ema_model), self.get_params(self.model)
        ):
            self.inplace_copy(model_param.data, ema_param.data)

        for (_, ema_buffer), (_, model_buffer) in zip(
            self.get_buffers(self.ema_model), self.get_buffers(self.model)
        ):
            self.inplace_copy(model_buffer.data, ema_buffer.data)

    def update(self):
        step = self.step.item()
        self.step += 1

        if step == self.update_after_step:
            self.copy_model_to_ema()
            return

        if (step % self.update_every) != 0:
            return

        if self.init_done.item() is False:
            self.copy_model_to_ema()
            self.init_done.data.copy_(torch.tensor(True))

        self.update_moving_average(self.ema_model, self.model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if self.is_frozen:
            return

        for (name, model_param), (_, ema_param) in zip(
            self.get_params(current_model), self.get_params(ma_model)
        ):
            if name in self.ignore_names:
                continue

            if any((name.startswith(prefix) for prefix in self.ignore_startswith)):
                continue

            if name in self.no_ema_names:
                self.inplace_copy(ema_param.data, model_param.data)
                continue

            self.inplace_lerp(ema_param.data, model_param.data, 1.0 - self.current_decay)

        for (name, model_buffer), (_, ema_buffer) in zip(
            self.get_buffers(current_model), self.get_buffers(ma_model)
        ):
            if name in self.ignore_names:
                continue

            if any((name.startswith(prefix) for prefix in self.ignore_startswith)):
                continue

            if name in self.no_ema_names:
                self.inplace_copy(ema_buffer.data, model_buffer.data)
                continue

            self.inplace_lerp(ema_buffer.data, model_buffer.data, 1.0 - self.current_decay)

    def forward(self, *args, **kwargs):
        return self.ema_model.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
