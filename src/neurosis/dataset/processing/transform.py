import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..base import SampleType

logger = logging.getLogger(__name__)


class DataTransform(ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        input_keys: list[str] = [],
        output_keys: list[str] = [],
        **kwargs,
    ):
        if not hasattr(self, "name"):
            if name is not None:
                self.name = str(name)
            else:
                self.name = str(self.__class__.__name__)

        self.input_keys = input_keys
        self.output_keys = output_keys

        if len(kwargs) > 0:
            logger.warning(f"{name} superclass received unexpected kwargs:\n" + f"{kwargs}")

    def __call__(self, sample: SampleType, raw) -> SampleType:
        return self.forward(sample)

    @abstractmethod
    def forward(self, sample: SampleType) -> SampleType:
        pass
