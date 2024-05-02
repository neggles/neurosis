import logging
from dataclasses import dataclass
from functools import partial
from os import getpid
from pathlib import Path
from socket import gethostname
from typing import Any, Callable, Iterable, Optional, Union
from warnings import warn

import torch.autograd.profiler as prof
from lightning.pytorch import LightningModule
from lightning.pytorch.profilers.profiler import Profiler
from torch import Tensor, nn
from torch._C._profiler import _ExperimentalConfig  # type: ignore
from torch.profiler import (
    ProfilerAction,
    ProfilerActivity,
    profile as kineto_profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.hooks import RemovableHandle
from typing_extensions import override

from neurosis.utils.system import get_next_dir

logger = logging.getLogger(__name__)


@dataclass
class ProfilerSchedule:
    skip_first: int = 0
    wait: int = 3
    warmup: int = 1
    active: int = 2
    repeat: int = 1

    def __post_init__(self) -> None:
        self.skip_first = max(self.skip_first, 0)
        self.wait = max(self.wait, 1)
        self.warmup = max(self.warmup, 1)
        self.active = max(self.active, 1)
        self.repeat = max(self.repeat, 1)

    def __call__(self) -> Callable[[int], ProfilerAction]:
        return self.schedule()

    def schedule(self) -> Callable[[int], ProfilerAction]:
        return schedule(
            skip_first=self.skip_first,
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
        )


@dataclass
class KinetoProfilerArgs:
    activities: Optional[list[ProfilerActivity]] = None
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = False
    with_modules: bool = True
    experimental_config: Optional[_ExperimentalConfig] = None

    def keys(self) -> Iterable[str]:
        return self.__match_args__

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.keys():
            val = getattr(self, key, None)
            return val if val is not None else default
        return default


class NeurosisProfiler(Profiler):
    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        create_run_dir: bool = False,
        capture_names: bool = False,
        capture_depth: int = -1,
        schedule: ProfilerSchedule = ProfilerSchedule(),
        profiler_kwargs: KinetoProfilerArgs = KinetoProfilerArgs(),
        verbose: bool = False,
    ):
        dirpath = Path(dirpath) if dirpath is not None else None
        if create_run_dir:
            if dirpath is None:
                raise ValueError("dirpath must be provided if create_run_dir=True")
            dirpath = get_next_dir(dirpath, "profile", sep="_")

        super().__init__(dirpath=dirpath, filename=filename)

        self._capture_names = capture_names
        self._capture_depth = capture_depth
        self._schedule = schedule
        self._profiler_kwargs = profiler_kwargs
        self._use_gzip = True  # hardcoded since I can't see a reason to *not* use gzip (yet?)
        self._verbose = verbose

        if self._capture_names is True and self._capture_depth < 0:
            warn("Capturing all module names can result in 10-100GB trace files! Good luck...", UserWarning)

        self.profiler: Optional[kineto_profile] = None
        self.function_events: dict[str, prof.EventList] = {}

        self._lightning_module: Optional[LightningModule] = None  # set by ProfilerConnector
        self._recording_map: dict[str, prof.record_function] = {}
        self._name_recorder: Optional[RecordModuleNames] = None

    @override
    def setup(self, stage: str, local_rank: Optional[int] = None, log_dir: Optional[str] = None) -> None:
        super().setup(stage=stage, local_rank=local_rank, log_dir=log_dir)
        if not self.filename:
            self.filename = f"{gethostname()}-{self.local_rank}-{getpid()}"

        if stage == "fit":
            if self.profiler:
                self._delete_profiler(stage)
            self._init_profiler(stage)
            profiler = self.profiler.__enter__()
            if profiler is not None:
                self.profiler = profiler

    @override
    def teardown(self, stage: Optional[str]) -> None:
        # clean up recording map
        for k in self._recording_map.keys():
            self.stop(k)
        self._recording_map = {}

        # exit profiler and save events
        self._delete_profiler(stage)

        super().teardown(stage=stage)

    @override
    def start(self, action_name: str) -> None:
        # logger.debug(f"action {action_name} start")
        if action_name is None:
            return

        if self._capture_names is True and self._lightning_module is not None and self._name_recorder is None:
            self._name_recorder = RecordModuleNames(self._lightning_module, max_depth=self._capture_depth)
            self._name_recorder.setup()

        if self.profiler is not None and action_name not in self._recording_map:
            # Add [pl][profile] in name for pytorch profiler to recognize
            recording = prof.record_function("[pl][profile]" + action_name)
            recording = recording.__enter__()
            self._recording_map[action_name] = recording

    @override
    def stop(self, action_name: str) -> None:
        # logger.debug(f"action {action_name} stop")

        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            _ = self._recording_map.pop(action_name)

        if self.profiler is None:
            return  # placed after the recording map check to ensure it's cleaned up

        if action_name.endswith("run_training_batch"):
            self.profiler.on_trace_ready = self._get_trace_handler(action_name=action_name)
            self.profiler.step()

    @override
    def summary(self) -> str:
        self._delete_profiler()

        function_events: prof.EventList = self.function_events.get(self._stage, None)
        if function_events is None:
            return ""

        data = function_events.key_averages()
        table = data.table(sort_by="cuda_time_total", row_limit=25)

        recorded_stats = {"records": table}
        return self._stats_to_str(recorded_stats)

    def _get_trace_handler(self, action_name: Optional[str] = None) -> Callable:
        return tensorboard_trace_handler(
            dir_name=str(self.dirpath.resolve()),
            worker_name=self._prepare_filename(action_name=action_name, extension=""),
            use_gzip=self._use_gzip,
        )

    def _init_profiler(self, stage: Optional[str] = None) -> None:
        stage = stage or self._stage
        if stage is None:
            return

        if self.profiler is None:
            logger.info(f"Initializing profiler for {stage}")
            if self._profiler_kwargs.experimental_config is None:
                self._profiler_kwargs.experimental_config = _ExperimentalConfig(verbose=self._verbose)
            self.profiler = kineto_profile(
                schedule=self._schedule(),
                on_trace_ready=self._get_trace_handler(),
                **self._profiler_kwargs,
            )

    def _delete_profiler(self, stage: Optional[str] = None) -> None:
        stage = stage or self._stage
        if stage is None:
            return

        # clean up recording map
        if self._name_recorder is not None:
            self._name_recorder.teardown()
            self._name_recorder = None

        # exit profiler and save events
        if self.profiler is not None:
            logger.info(f"Closing profiler for {stage}")
            # clean up recording map
            for k in self._recording_map:
                self._recording_map[k].__exit__(None, None, None)
            self._recording_map = {}

            # save events
            self.profiler.__exit__(None, None, None)
            self.function_events[stage] = self.profiler.events()
            self.profiler = None


class RecordModuleNames:
    """While profiling autograd operations, this class will add labels for module names around the forward function.

    The Lightning PyTorch Profiler will activate this feature automatically. It can be deactivated as follows:

    Example::
        from lightning.pytorch.profilers import PyTorchProfiler

        profiler = PyTorchProfiler(record_module_names=False)
        Trainer(profiler=profiler)

    It can be used outside of Lightning as follows:

    Example::
        from lightning.pytorch import Trainer, seed_everything

        with RecordModuleNames(model):
            out = model(batch)

    """

    def __init__(self, model: nn.Module, max_depth: int = -1) -> None:
        self._model = model
        self._max_depth = max_depth
        self._records: dict[str, record_function] = {}
        self._handles: dict[str, list[RemovableHandle]] = {}

    def _start_recording_forward(self, _: nn.Module, input: Tensor, record_name: str) -> Tensor:
        record = record_function("[mod]" + record_name)
        record.__enter__()
        self._records[record_name] = record
        return input

    def _stop_recording_forward(self, _: nn.Module, __: Tensor, output: Tensor, record_name: str) -> Tensor:
        self._records[record_name].__exit__(None, None, None)
        del self._records[record_name]
        return output

    def setup(self) -> None:
        if len(self._handles) > 0:
            return  # already set up, don't do it again

        for module_name, module in self._model.named_modules():
            if module_name:
                if self._max_depth > 0 and len(module_name.split(".")) > self._max_depth:
                    continue  # skip modules that are too deep

                module_package = type(module).__module__.removeprefix("torch.nn.")
                record_name = f"{module_name}: {module_package}.{type(module).__name__}"

                pre_forward_handle = module.register_forward_pre_hook(
                    partial(self._start_recording_forward, record_name=record_name)
                )
                post_forward_handle = module.register_forward_hook(
                    partial(self._stop_recording_forward, record_name=record_name)
                )

                self._handles[module_name] = [pre_forward_handle, post_forward_handle]

    def teardown(self) -> None:
        for handles in self._handles.values():
            for h in handles:
                h.remove()
        self._handles = {}
