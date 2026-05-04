from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from PIL import Image

class BaseSlideBackend(ABC):

    @abstractmethod
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        pass

    @property
    @abstractmethod
    def level_downsamples(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def level_dimensions(self) -> List[Tuple[int, int]]:
        pass

    @property
    def associated_images(self):
        return {}

    @property
    @abstractmethod
    def level_count(self) -> int:
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

    def get_best_level_for_downsample(self, ratio: float) -> int:
        diffs = [abs(ds - ratio) for ds in self.level_downsamples]
        return diffs.index(min(diffs))

