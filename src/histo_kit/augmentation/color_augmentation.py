from PIL import Image, ImageEnhance
import numpy as np
import random
from ..augmentation.base import Transform

class SaltAndPepper(Transform):
    def __init__(self, amount_range=(0.01, 0.05), prob=1.0):
        super().__init__(prob)
        self.amount_range = amount_range

    def apply(self, img: Image.Image) -> Image.Image:
        amount = random.uniform(*self.amount_range)
        arr = np.array(img)
        num_salt = np.ceil(amount * arr.size * 0.5)
        num_pepper = np.ceil(amount * arr.size * 0.5)

        # Salt
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape[:2]]
        arr[coords[0], coords[1]] = 255

        # Pepper
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape[:2]]
        arr[coords[0], coords[1]] = 0

        return Image.fromarray(arr)

class GaussianNoise(Transform):
    def __init__(self, mean_range=(0.0, 6.0), std_range=(10.0, 30.0), prob=1.0):
        super().__init__(prob)
        self.mean_range = mean_range
        self.std_range = std_range

    def apply(self, img: Image.Image) -> Image.Image:
        mean = random.uniform(*self.mean_range)
        std = random.uniform(*self.std_range)
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(mean, std, arr.shape)
        arr += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class ColorJitter(Transform):
    def __init__(self, brightness_range=(0, 0.3), contrast_range=(0, 0.3),
                 saturation_range=(0, 0.3), hue_range=(0, 0.1), prob=1.0):
        super().__init__(prob)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range

    def apply(self, img: Image.Image) -> Image.Image:
        # Brightness
        if self.brightness_range[1] > 0:
            factor = 1 + random.uniform(*self.brightness_range)
            img = ImageEnhance.Brightness(img).enhance(factor)
        # Contrast
        if self.contrast_range[1] > 0:
            factor = 1 + random.uniform(*self.contrast_range)
            img = ImageEnhance.Contrast(img).enhance(factor)
        # Saturation
        if self.saturation_range[1] > 0:
            factor = 1 + random.uniform(*self.saturation_range)
            img = ImageEnhance.Color(img).enhance(factor)
        # Hue
        if self.hue_range[1] > 0:
            hsv = img.convert("HSV")
            arr = np.array(hsv)
            arr[..., 0] = (arr[..., 0].astype(int) + int(random.uniform(
                self.hue_range[0] * 255, self.hue_range[1] * 255))) % 255
            img = Image.fromarray(arr, "HSV").convert("RGB")
        return img
