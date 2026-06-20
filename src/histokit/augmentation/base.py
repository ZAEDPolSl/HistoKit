import random
import numpy as np
from abc import ABC, abstractmethod

class Transform(ABC):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.prob:
            return self.apply(img)
        return img

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        pass

class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.prob:
            transform = random.choice(self.transforms)
            return transform(img)

        return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            img = transform(img)

        return img