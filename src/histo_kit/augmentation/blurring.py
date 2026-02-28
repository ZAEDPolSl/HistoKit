from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import random
import io
from ..augmentation.base import Transform

class GaussianBlur(Transform):
    def __init__(self, radius_range=(0.5, 3.0), prob=1.0):
        super().__init__(prob)
        self.radius_range = radius_range

    def apply(self, img: Image.Image) -> Image.Image:
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius))

class MedianBlur(Transform):
    def __init__(self, size_range=(3, 7), prob=1.0):
        super().__init__(prob)
        self.size_range = size_range

    def apply(self, img: Image.Image) -> Image.Image:
        size = random.randint(*self.size_range)
        return img.filter(ImageFilter.MedianFilter(size))

class MotionBlur(Transform):
    def __init__(self, degree_range=(5, 15), angle_range=(0, 360), prob=1.0):
        super().__init__(prob)
        self.degree_range = degree_range
        self.angle_range = angle_range

    def apply(self, img: Image.Image) -> Image.Image:
        import cv2
        degree = random.randint(*self.degree_range)
        angle = random.uniform(*self.angle_range)
        img_cv = np.array(img)
        k = np.zeros((degree, degree))
        k[int((degree - 1) / 2), :] = np.ones(degree)
        k = k / degree
        M = cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1)
        k = cv2.warpAffine(k, M, (degree, degree))
        img_cv = cv2.filter2D(img_cv, -1, k)
        return Image.fromarray(img_cv)

class JPEGCompression(Transform):
    def __init__(self, quality_range=(10, 50), prob=1.0):
        super().__init__(prob)
        self.quality_range = quality_range

    def apply(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)