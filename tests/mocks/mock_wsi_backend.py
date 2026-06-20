from typing import Tuple, List, Optional
from PIL import Image, ImageDraw
from histokit.slide.backends import BaseSlideBackend

class MockWSIBackend(BaseSlideBackend):
    def __init__(self):

        self._level_downsamples = [1.0, 2.0, 4.0]
        self._level_dimensions = [
            (4000, 4000),
            (2000, 2000),
            (1000, 1000),
        ]

        self._mag = 20
        self._mpp = 0.5
        self._properties = {}
        self.last_call = None

        self._levels = []
        self._create_levels()

    def _create_levels(self):
        center_l0 = (2000, 2000)
        radius_l0 = 400

        for level, downsample in enumerate(self._level_downsamples):
            size = self._level_dimensions[level]
            img = Image.new("RGBA", size, color=(255, 255, 255, 255))
            draw = ImageDraw.Draw(img)

            cx = center_l0[0] / downsample
            cy = center_l0[1] / downsample
            r = radius_l0 / downsample

            draw.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                fill=(0, 0, 255, 255),
            )

            dot_r = 1

            draw.ellipse(
                (cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r),
                fill=(255, 0, 0, 255),
            )

            self._levels.append(img)

    def read_region(self, location, level, size):
        self.last_call = {
            "location": location,
            "level": level,
            "size": size,
        }

        x, y = location
        w, h = size

        level_img = self._levels[level]

        downsample = self._level_downsamples[level]
        x_level = int(x / downsample)
        y_level = int(y / downsample)

        return level_img.crop((x_level, y_level, x_level + w, y_level + h))

    @property
    def level_downsamples(self) -> List[float]:
        return self._level_downsamples

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        return self._level_dimensions

    @property
    def level_count(self) -> int:
        return len(self._level_downsamples)

    @property
    def properties(self) -> dict:
        return self._properties

    @property
    def mag(self) -> Optional[float]:
        return self._mag

    @property
    def mpp(self) -> Optional[float]:
        return self._mpp