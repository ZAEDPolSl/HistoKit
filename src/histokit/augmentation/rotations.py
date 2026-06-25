import numpy as np
import random
from ..augmentation.base import Transform

class RandomFlip(Transform):
    """Randomly flip an image horizontally and/or vertically.

    This transform independently applies horizontal and vertical flips with
    probability ``1.0``.

    Parameters
    ----------
    prob : float, default=1.0
        Probability of applying the transform.

    Notes
    -----
    The returned array is made contiguous using ``np.ascontiguousarray``.
    """
    def __init__(self, prob=1.0):
        super().__init__(prob)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply random horizontal and vertical flips.

        Parameters
        ----------
        img : np.ndarray
            Input image with shape ``(H, W)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Flipped image with the same shape and dtype as ``img``.
        """
        if random.random() < self.prob:
            img = np.fliplr(img)

        if random.random() < self.prob:
            img = np.flipud(img)

        return np.ascontiguousarray(img)


class RandomRotation(Transform):
    """Randomly rotate an image by 90-degree increments.

    This transform rotates the input image by a random multiple of 90 degrees,
    selected from the set {0, 1, 2, 3}.

    Parameters
    ----------
    prob : float, default=1.0
        Probability of applying the transform.

    Notes
    -----
    The returned array is made contiguous using ``np.ascontiguousarray``.
    """
    def __init__(self, prob=1.0):
        super().__init__(prob)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply a random 90-degree rotation.

        Parameters
        ----------
        img : np.ndarray
            Input image with shape ``(H, W)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Rotated image. The output has the same dtype as ``img``. Its shape
            may have swapped height and width when rotated by 90 or 270 degrees.
        """
        k = random.randint(0, 3)
        return np.ascontiguousarray(np.rot90(img, k))