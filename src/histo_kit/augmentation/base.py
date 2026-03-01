import random
from PIL import Image
from abc import ABC, abstractmethod

class Transform(ABC):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            return self.apply(img)
        return img

    @abstractmethod
    def apply(self, img: Image.Image) -> Image.Image:
        pass

class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transform = transforms
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.prob:
            t = random.choice(self.transform)
            return t(img)
        return img

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        for t in self.transforms:
            img = t(img)
        return img
