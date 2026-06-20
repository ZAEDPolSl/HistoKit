from PIL import Image
from typing import Optional, Dict, Any
from PIL.ExifTags import TAGS
from .base import BaseSlideBackend

class PILBackend(BaseSlideBackend):

    def __init__(self, data: str | Image.Image):

        if isinstance(data, Image.Image):
            self._slide = data
        else:
            self._slide = Image.open(data)
        self._properties = self._get_metadata(data)
        self._properties["PIL_mode"] = self._slide.mode

    def read_region(self, location, level, size) -> Image.Image:
        x0, y0 = location
        w, h = size
        return self._slide.crop((x0, y0, x0 + w, y0 + h))

    @property
    def level_downsamples(self):
        return [1.0]

    @property
    def level_dimensions(self):
        return [self._slide.size]

    @property
    def level_count(self):
        return 1

    @property
    def properties(self):
        return self._properties

    @property
    def mag(self) -> Optional[float]:
        # TODO: Estimate magnification
        return None

    @property
    def mpp(self) -> Optional[float]:
        # TODO: Estimate mpp
        return None

    @staticmethod
    def _get_metadata(path: str) -> Dict[str, Any]:
        with Image.open(path) as img:
            exif_data = img.getexif()
            return {
                TAGS.get(tag, tag): value
                for tag, value in exif_data.items()
            }

