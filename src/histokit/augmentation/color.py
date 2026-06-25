import cv2
import numpy as np
import random
from .base import Transform


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
