from PIL import Image, ImageOps
import random
from ..augmentation.base import Transform


class RandomFlip(Transform):
    def __init__(self, prob=0.5):
        super().__init__(prob)

    def apply(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        if random.random() < 0.5:
            img = ImageOps.flip(img)
        return img

class RandomRotation(Transform):
    def __init__(self, angles=(0, 90, 180, 270), prob=0.5):
        super().__init__(prob)
        self.angles = angles

    def apply(self, img: Image.Image) -> Image.Image:
        angle = random.choice(self.angles)
        return img.rotate(angle, expand=True)
