from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union, Sequence

import numpy as np

from src.histo_kit.slide.bbox import BBox, BBoxMode


class BaseMaskBackend(ABC):

    @property
    @abstractmethod
    def mask_dimensions(self) -> List[Tuple[int, int]]:
        pass

    @property
    @abstractmethod
    def mask_count(self) -> int:
        pass

    @property
    @abstractmethod
    def mask_array(self) -> List[np.ndarray]:
        pass

    @property
    @abstractmethod
    def properties(self) -> dict:
        pass

    @property
    @abstractmethod
    def mag(self) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def mpp(self) -> Optional[float]:
        pass

