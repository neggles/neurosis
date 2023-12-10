import resource
from dataclasses import dataclass
from typing import Optional, Union

import psutil
import torch


class GPUMemoryUsage:
    def __init__(self, device: Union[str, torch.device] = "cuda:0"):
        self.device = torch.device(device)  # ensure device is a torch.device
        self.free = -1
        self.total = -1

    @property
    def update(self) -> None:
        if torch.cuda.is_available():
            self.free, self.total = torch.cuda.mem_get_info(self.device)
        else:
            self.free, self.total = -1, -1

    @property
    def used(self) -> int:
        return self.total - self.free

    def __str__(self):
        return "GPU{:d}: (U: {:,}MiB F: {:,}MiB T: {:,}MiB)".format(
            self.device.index, self.used >> 20, self.free >> 20, self.total >> 20
        )


class TorchGPUMemoryUsage:
    reserved: int
    reserved_max: int
    used: int
    used_max: int

    @classmethod
    def now(cls, device=None):
        if torch.cuda.is_available():
            stats = torch.cuda.memory.memory_stats(device)
            return cls(
                stats.get("reserved_bytes.all.current", 0),
                stats.get("reserved_bytes.all.peak", 0),
                stats.get("allocated_bytes.all.current", 0),
                stats.get("allocated_bytes.all.peak", 0),
            )
        else:
            return None

    def __str__(self):
        return "TORCH: (R: {:,}MiB/{:,}MiB, A: {:,}MiB/{:,}MiB)".format(
            self.reserved >> 20, self.reserved_max >> 20, self.used >> 20, self.used_max >> 20
        )


class CPUMemoryUsage:
    maxrss_kibibytes: int
    free: int

    @classmethod
    def now(cls):
        maxrss = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        )
        vmem = psutil.virtual_memory()
        return cls(maxrss, vmem.free)

    def __str__(self):
        return "CPU: (maxrss: {:,}MiB F: {:,}MiB)".format(self.maxrss_kibibytes >> 10, self.free >> 20)


@dataclass
class MemoryUsageInfo:
    cpu: CPUMemoryUsage
    gpu: Optional[GPUMemoryUsage]
    torch: Optional[TorchGPUMemoryUsage]

    @classmethod
    def now(cls):
        gpu_info = torch_info = None
        try:
            gpu_info = GPUMemoryUsage()
            torch_info = TorchGPUMemoryUsage()
        except AssertionError:
            pass
        return cls(CPUMemoryUsage.now(), gpu_info, torch_info)

    def __str__(self):
        return " ".join(str(item) for item in self if item)
