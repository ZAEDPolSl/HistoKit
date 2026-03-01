import scipy.ndimage as ndi
from .algorithms.remove_small_objects import remove_small_objects
from ...utils.matlab2python import get_strel_disk
from abc import ABC, abstractmethod
import numpy as np

class PostProcessStep(ABC):

    @abstractmethod
    def __call__(self, mask: np.ndarray, config) -> np.ndarray:
        pass

class Opening(PostProcessStep):
    def __call__(self, mask, config):
        SE = get_strel_disk(config.open_disk_radius)
        return ndi.binary_opening(mask, SE)

class FillHoles(PostProcessStep):
    def __call__(self, mask, config):
        if config.fill_holes:
            return ndi.binary_fill_holes(mask)
        return mask

class RemoveSmallObjects(PostProcessStep):
    def __call__(self, mask, config):
        if config.remove_small_objects:
            return remove_small_objects(mask)
        return mask