import cv2
import numpy as np
import random
from .base import Transform

class SaltAndPepper(Transform):
    def __init__(self, amount_range=(0.01, 0.05), prob=1.0):
        super().__init__(prob)
        self.amount_range = amount_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        arr = img.copy()
        amount = random.uniform(*self.amount_range)

        h, w = arr.shape[:2]
        num_pixels = int(amount * h * w)

        num_salt = num_pixels // 2
        num_pepper = num_pixels - num_salt

        ys = np.random.randint(0, h, num_salt)
        xs = np.random.randint(0, w, num_salt)
        arr[ys, xs] = 255

        ys = np.random.randint(0, h, num_pepper)
        xs = np.random.randint(0, w, num_pepper)
        arr[ys, xs] = 0

        return arr


class GaussianNoise(Transform):
    def __init__(self, mean_range=(0.0, 6.0), std_range=(10.0, 30.0), prob=1.0):
        super().__init__(prob)
        self.mean_range = mean_range
        self.std_range = std_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        mean = random.uniform(*self.mean_range)
        std = random.uniform(*self.std_range)

        arr = img.astype(np.float32)
        noise = np.random.normal(mean, std, arr.shape)

        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr


class ColorJitter(Transform):
    def __init__(
        self,
        brightness_range=(0, 0.3),
        contrast_range=(0, 0.3),
        saturation_range=(0, 0.3),
        hue_range=(0, 0.1),
        prob=1.0,
    ):
        super().__init__(prob)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        arr = img.astype(np.float32)

        # Brightness
        if self.brightness_range[1] > 0:
            factor = 1.0 + random.uniform(*self.brightness_range)
            arr = arr * factor

        # Contrast
        if self.contrast_range[1] > 0:
            factor = 1.0 + random.uniform(*self.contrast_range)
            mean = arr.mean(axis=(0, 1), keepdims=True)
            arr = (arr - mean) * factor + mean

        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Saturation / Hue
        if self.saturation_range[1] > 0 or self.hue_range[1] > 0:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)

            if self.saturation_range[1] > 0:
                factor = 1.0 + random.uniform(*self.saturation_range)
                hsv[..., 1] *= factor

            if self.hue_range[1] > 0:
                shift = random.uniform(
                    self.hue_range[0] * 179,
                    self.hue_range[1] * 179,
                )
                hsv[..., 0] = (hsv[..., 0] + shift) % 180

            hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
            hsv = hsv.astype(np.uint8)

            arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return arr
