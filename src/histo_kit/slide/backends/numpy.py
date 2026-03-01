import numpy as np
from typing import Optional
from PIL import Image
from .base import BaseSlideBackend

class NumpyBackend(BaseSlideBackend):

    def __init__(self, data: np.array):
        self._slide = data
        self._properties = {"mode": self._detect_image_mode(data)}

    def read_region(self, location, level, size) -> Image.Image:
        x0, y0 = location
        w, h = size
        region = self._slide[y0:y0 + h, x0:x0 + w, ...]
        return Image.fromarray(region).convert(self._properties["mode"])

    @property
    def level_downsamples(self):
        return [1.0]

    @property
    def level_dimensions(self):
        if self._slide.ndim == 2:
            h, w = self._slide.shape
        elif self._slide.ndim == 3:
            h, w, _ = self._slide.shape
        else:
            raise ValueError(f"Unsupported slide shape: {self._slide.shape}")
        return [(w, h)]

    @property
    def level_count(self):
        return 1

    @property
    def properties(self):
        return self._properties

    @property
    def mag(self) -> Optional[float]:
        return None

    @property
    def mpp(self) -> Optional[float]:
        return None

    def _detect_image_mode(self, data: np.ndarray):
        if data.ndim == 2:

            if data.dtype == bool:
                return "1" # binary (1-bit) - boolean

            if data.dtype == np.uint8:
                unique_vals = np.unique(data)

                if set(unique_vals).issubset({0, 1}) or set(unique_vals).issubset({0, 255}):
                    return "1" # binary (1-bit) - either 0/1 or 0/255

                return "L" # grayscale uint8 (2D)

            if np.issubdtype(data.dtype, np.floating):
                return "F" # float32 (2D)

            if data.dtype == np.int32:
                return "I" # int32 (2D)

            raise NotImplementedError(f"Unsupported dtype for 2D data: {data.dtype}")

        elif data.ndim == 3:
            channels = data.shape[2]

            if channels == 3:
                return "RGB"
            elif channels == 4:
                return "RGBA"

            raise NotImplementedError(f"Unsupported number of channels: {channels}")

        else:
            raise NotImplementedError(f"Unsupported data shape: {data.shape}")