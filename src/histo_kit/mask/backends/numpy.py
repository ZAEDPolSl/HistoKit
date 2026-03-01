from typing import Optional, List, Sequence, Union
import numpy as np
from .base import BaseMaskBackend

class NumpyMaskBackend(BaseMaskBackend):

    def __init__(self,
                 mask: np.ndarray | Sequence[np.array],
                 properties: dict = None,
                 mag: float = None,
                 mpp: float = None):

        self._mask_array = self._convert_masks(mask)
        self._properties = properties if properties is not None else {}
        self._mag = mag if mag is not None else self._properties.get("mag", None)
        self._mpp = mpp if mpp is not None else self._properties.get("mpp", None)


    @staticmethod
    def _convert_masks(mask: Union[List[np.ndarray]]) -> List[np.ndarray]:

        if isinstance(mask, np.ndarray):
            if mask.ndim != 2:
                raise ValueError(f"Single mask must be 2D, got shape {mask.shape}")
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            return [mask]

        elif isinstance(mask, (list, tuple)):
            converted = []
            for i, m in enumerate(mask):
                if not isinstance(m, np.ndarray):
                    raise TypeError(f"Mask at index {i} is not a numpy array, got {type(m)}")
                if m.ndim != 2:
                    raise ValueError(f"Mask at index {i} must be 2D, got shape {m.shape}")
                if m.dtype != np.uint8:
                    m = m.astype(np.uint8)
                converted.append(m)
            return converted

        else:
            raise TypeError(f"mask must be np.ndarray or list of ndarrays, got {type(mask)}")


    @property
    def mask_dimensions(self) -> list[tuple[int, ...]]:
        dims = []

        for i, m in enumerate(self._mask_array):
            dims.append(m.shape)
        return dims

    @property
    def mask_count(self) -> int:
        return len(self._mask_array)

    @property
    def properties(self) -> dict:
        return self._properties

    @property
    def mag(self) -> Optional[float]:
        return self._mag

    @property
    def mpp(self) -> Optional[float]:
        return self._mpp

    @property
    def mask_array(self) -> List[np.ndarray]:
        return self._mask_array