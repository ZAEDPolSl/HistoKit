import numpy as np
import random
from ..augmentation.base import Transform

class RandomFlip(Transform):
    def __init__(self, prob=0.5):
        super().__init__(prob)

    def apply(self, img: np.ndarray) -> np.ndarray:

        if random.random() < 0.5:
            img = np.fliplr(img)

        if random.random() < 0.5:
            img = np.flipud(img)

        return np.ascontiguousarray(img)


class RandomRotation(Transform):
    def __init__(self, prob=0.5):
        super().__init__(prob)

    def apply(self, img: np.ndarray) -> np.ndarray:
        k = random.randint(0, 3)
        return np.ascontiguousarray(np.rot90(img, k))