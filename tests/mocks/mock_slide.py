import numpy as np
from PIL import Image

class MockSlide:
    def __init__(self, backend):
        self.backend = backend

    def read_object(self, bbox, mag=None, mpp=None, level=None):
        if level is None:
            level = 0

        img = self.backend._levels[level]

        x0, y0 = int(bbox.x0), int(bbox.y0)
        x1, y1 = int(bbox.x1), int(bbox.y1)

        return img.crop((x0, y0, x1, y1))