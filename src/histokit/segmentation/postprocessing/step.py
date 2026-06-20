from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from skimage.morphology import disk
from .algorithms.remove_small_regions import remove_small_regions
import scipy.ndimage as ndi

class PostProcessStep(ABC):
    """
        Abstract base class for post-processing operations applied to masks.

        Each post-processing step takes a binary mask and a configuration
        object as input and returns a processed mask.
    """

    @abstractmethod
    def __call__(self, mask: np.ndarray) -> np.ndarray:
        ...

    def get_config(self):
        ...


class RemoveSmallRegions(PostProcessStep):

    def __init__(self, thr_area: int = None):
        self.thr_area = thr_area

    def __call__(self, mask):
        return remove_small_regions(mask, thr_area=self.thr_area)

    def get_config(self) -> dict[str, Any]:
        return {
            "name": "RemoveSmallRegions",
            "params": {
                "thr_area": self.thr_area,
            },
        }


class Opening(PostProcessStep):
    def __init__(self, disk_radius: int = 2):
        self.disk_radius = disk_radius

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        se = disk(self.disk_radius)
        return ndi.binary_opening(mask, structure=se)

    def get_config(self) -> dict[str, Any]:
        return {
            "name": "Opening",
            "params": {
                "disk_radius": self.disk_radius,
            },
        }

class FillHoles(PostProcessStep):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return mask

        return ndi.binary_fill_holes(mask)

    def get_config(self) -> dict[str, Any]:
        return {
            "name": "FillHoles",
            "params": {
                "enabled": self.enabled,
            },
        }