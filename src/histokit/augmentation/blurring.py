import cv2
import random
import numpy as np
from ..augmentation.base import Transform

class GaussianBlur(Transform):
    def __init__(self, sigma_range=(0.5, 3.0), prob=1.0):
        super().__init__(prob)
        self.sigma_range = sigma_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        sigma = random.uniform(*self.sigma_range)
        return cv2.GaussianBlur(img, (0, 0), sigma)


class MedianBlur(Transform):
    def __init__(self, size_range=(3, 7), prob=1.0):
        super().__init__(prob)
        self.size_range = size_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        size = random.randint(*self.size_range)

        if size % 2 == 0:
            size += 1

        return cv2.medianBlur(img, size)


class MotionBlur(Transform):
    def __init__(
        self,
        degree_range=(5, 15),
        angle_range=(0, 360),
        prob=1.0,
    ):
        super().__init__(prob)
        self.degree_range = degree_range
        self.angle_range = angle_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        degree = random.randint(*self.degree_range)
        angle = random.uniform(*self.angle_range)

        kernel = np.zeros((degree, degree), dtype=np.float32)
        kernel[(degree - 1) // 2, :] = 1.0
        kernel /= degree

        M = cv2.getRotationMatrix2D(
            (degree / 2 - 0.5, degree / 2 - 0.5),
            angle,
            1.0,
        )

        kernel = cv2.warpAffine(
            kernel,
            M,
            (degree, degree),
        )

        return cv2.filter2D(img, -1, kernel)


class JPEGCompression(Transform):
    def __init__(
        self,
        quality_range=(10, 50),
        prob=1.0,
    ):
        super().__init__(prob)
        self.quality_range = quality_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        quality = random.randint(*self.quality_range)

        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

        if not success:
            return img

        decoded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        return cv2.cvtColor(
            decoded,
            cv2.COLOR_BGR2RGB,
        )