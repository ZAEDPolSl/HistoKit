from openslide import OpenSlide
from typing import Optional
from PIL import Image
from .base import BaseSlideBackend


class OpenSlideBackend(BaseSlideBackend):

    def __init__(self, filepath: str):
        self._slide = OpenSlide(filepath)

    def read_region(self, location, level, size) -> Image.Image:
        return self._slide.read_region(location, level, size)

    @property
    def associated_images(self):
        return dict(self._slide.associated_images)

    @property
    def level_downsamples(self):
        return list(self._slide.level_downsamples)

    @property
    def level_dimensions(self):
        return list(self._slide.level_dimensions)

    @property
    def level_count(self):
        return self._slide.level_count

    @property
    def properties(self):
        return dict(self._slide.properties)

    @property
    def mag(self) -> Optional[float]:
        value = self._slide.properties.get("openslide.objective-power")
        return float(value) if value else None

    @property
    def mpp(self) -> Optional[float]:
        value = self._slide.properties.get("openslide.mpp-x")
        return float(value) if value else None

    def get_best_level_for_downsample(self, ratio: float) -> int:
        return self._slide.get_best_level_for_downsample(ratio)