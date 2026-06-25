import warnings
import numpy as np
from .base import Transform

class StainNormalizationTransform(Transform):
    """Apply stain normalization as an augmentation transform.

    This transform wraps a stain normalizer object and applies its
    ``transform`` method to the input image.

    Parameters
    ----------
    normalizer : object
        Fitted stain normalizer object. The object is expected to implement
        a ``transform(img)`` method that accepts an RGB image as a NumPy array.
    prob : float, default=1.0
        Probability of applying the transform.

    Notes
    -----
    If stain normalization fails, a warning is emitted and an empty white image
    with the same shape and dtype as the input image is returned.
    """

    def __init__(self, normalizer, prob=1.0):
        super().__init__(prob)
        self.normalizer = normalizer

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply stain normalization to an image.

        Parameters
        ----------
        img : np.ndarray
            Input RGB image with shape ``(H, W, 3)`` and pixel values usually
            in ``[0, 255]``.

        Returns
        -------
        np.ndarray
            Stain-normalized image. If normalization fails, returns a white
            image with the same shape and dtype as ``img``.

        Warns
        -----
        RuntimeWarning
            If the normalizer fails to transform the input image.
        """
        try:
            img_norm = self.normalizer.transform(img)
            return img_norm

        except Exception:
            warnings.warn(
                "Stain normalization failed. Returning an empty white image (255, 255, 255) "
                "This may be due to the image being too small, empty, or having "
                "very low contrast. Consider adjusting the parameters of the "
                "normalizer or using a different normalizer.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.ones_like(img) * 255