import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataTransform(ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs,
    ):
        if not hasattr(self, "name"):
            if name is not None:
                self.name = str(name)
            else:
                self.name = str(self.__class__.__name__)

        if len(kwargs) > 0:
            logger.warning(f"{name} superclass received unexpected kwargs:\n" + f"{kwargs}")

    def __call__(
        self,
        sample: pd.DataFrame,
    ) -> pd.DataFrame:
        return self.forward(sample)

    @abstractmethod
    def forward(
        self,
        sample: pd.DataFrame,
    ) -> pd.DataFrame:
        pass
