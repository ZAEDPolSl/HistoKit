import warnings
import numpy as np
from .base import Transform

class StainNormalizationTransform(Transform):
    def __init__(self, normalizer, prob=1.0):
        super().__init__(prob)
        self.normalizer = normalizer

    def apply(self, img: np.ndarray) -> np.ndarray:
        try:
            img_norm = self.normalizer.transform(img)
            return img_norm
        except:
            warnings.warn(
                "Empty mask computed. Returning empty image. This may be due to the image being too small or having very low contrast. " \
                "Consider adjusting the parameters of the normalizer or using a different normalizer.",
                RuntimeWarning,
            )
            return np.ones_like(img)