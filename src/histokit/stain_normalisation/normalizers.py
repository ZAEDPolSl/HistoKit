import cv2
import numpy as np
from abc import abstractmethod, ABC
from .extractors import BaseExtractor, MacenkoExtractor, VahadaneExtractor
from .utils import get_concentrations, od2rgb, is_rgb_uint8, get_tissue_mask


class BaseNormalizer(ABC):

    @abstractmethod
    def fit(self, target: np.ndarray):
        """
        Fit the underlying normalizer to a target image.

        Parameters
        ----------
        target : np.ndarray
            RGB uint8 target image.

        Returns
        -------
        StainingNormalizer
            Fitted normalizer instance (self).
        """
        pass

    @abstractmethod
    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Transform an input image to match the fitted target distribution.

        Parameters
        ----------
        img : np.ndarray
            RGB uint8 source image to be normalized.

        Returns
        -------
        np.ndarray
            Normalized RGB uint8 image.
        """
        pass


class StainMatrixNormalizer(BaseNormalizer):
    """
    Normalizer that matches stain matrices using an extractor.

    This normalizer uses a provided extractor (Macenko, Vahadane) to
    estimate stain matrices for source and target images, computes concentration
    percentiles and rescales source concentrations to match the target before
    reconstructing the normalized image.

    Parameters
    ----------
    extractor : BaseExtractor
        Extractor used to compute stain matrices for images.

    Attributes
    ----------
    extractor : BaseExtractor
        Provided extractor instance.
    target_mat : np.ndarray or None
        OD stain matrix estimated from the target image after fitting.
    maxC_target : np.ndarray or None
        Per-stain 99th percentile concentration values from the target image.
    stain_mat_target_rgb : np.ndarray or None
        RGB representation of the target stain matrix.
    """
    def __init__(self, extractor: BaseExtractor):
        self.extractor = extractor
        self.target_mat = None
        self.maxC_target = None
        self.stain_mat_target_rgb = None

    def fit(self, target):
        """
        Fit the normalizer using a target image.

        Parameters
        ----------
        target : np.ndarray
            RGB uint8 target image.

        Returns
        -------
        StainMatrixNormalizer
            Fitted normalizer.

        Raises
        ------
        AssertionError
            If `target` is not an RGB uint8 numpy array.
        """
        assert is_rgb_uint8(target), "Target should be RGB uint8 np.ndarray"

        self.target_mat = self.extractor.get_stain_matrix(target)
        target_concentrations = get_concentrations(target, self.target_mat)

        self.maxC_target = np.percentile(
            target_concentrations,
            99,
            axis=0
        ).reshape((1, 2))

        self.stain_mat_target_rgb = od2rgb(self.target_mat)

        return self

    def transform(self, img):
        """
        Transform `img` to match the fitted target stain appearance.

        Parameters
        ----------
        img : np.ndarray
            RGB uint8 source image to be normalized.

        Returns
        -------
        np.ndarray
            Normalized RGB uint8 image.

        Raises
        ------
        AssertionError
            If `img` is not an RGB uint8 numpy array.
        RuntimeError
            If the normalizer has not been fitted with a target.
        """
        assert is_rgb_uint8(img), "Image should be RGB uint8 np.ndarray"

        if self.target_mat is None:
            raise RuntimeError("Normalizer must be fitted before transform().")

        src_mat = self.extractor.get_stain_matrix(img)
        src_concentrations = get_concentrations(img, src_mat)

        maxC_source = np.percentile(
            src_concentrations,
            99,
            axis=0
        ).reshape((1, 2))

        src_concentrations *= self.maxC_target / maxC_source

        img_norm = 255 * np.exp(
            -np.dot(src_concentrations, self.target_mat)
        )

        return img_norm.reshape(img.shape).clip(0, 255).astype(np.uint8)


class ReinhardNormalizer(BaseNormalizer):
    """
    Reinhard colour normalizer.

    Perform colour normalization based on Reinhard et al. (2001). This approach
    matches mean and standard deviation of the L\*a\*b\* channels between source
    and target images.

    Notes
    -----
    This implementation uses a tissue mask (via :pyfunc:`get_tissue_mask`) to
    preserve background pixels from being altered.

    References
    ----------
    Reinhard, E., et al. "Color transfer between images." IEEE Computer
    graphics and applications 21.5 (2001): 34-41.
    """

    def __init__(self, lum_thr=1):
        """
        Initialize the Reinhard normalizer.

        Parameters
        ----------
        lum_thr : float, optional
            Luminosity threshold used by mask computation in some contexts
            (default is 1).
        """
        self.target_means = None
        self.target_stds = None
        self.lum_thr = lum_thr

    def fit(self, target):
        """
        Fit the normalizer using a target image by computing channel means and stds.

        Parameters
        ----------
        target : np.ndarray
            RGB uint8 target image.

        Returns
        -------
        ReinhardNormalizer
            Fitted normalizer.

        Raises
        ------
        AssertionError
            If `target` is not an RGB uint8 numpy array.
        """
        assert is_rgb_uint8(target), "Target should be RGB uint8 np.ndarray"
        self.target_means, self.target_stds = self.get_mean_std(target)
        return self

    def transform(self, img):
        """
        Transform `img` to match the target L\*a\*b\* statistics.

        Parameters
        ----------
        img : np.ndarray
            RGB uint8 source image to be normalized.

        Returns
        -------
        np.ndarray
            Normalized RGB uint8 image where background pixels are preserved.

        Raises
        ------
        AssertionError
            If `img` is not an RGB uint8 numpy array.
        RuntimeError
            If the normalizer has not been fitted.
        """
        assert is_rgb_uint8(img), "Image should be RGB uint8 np.ndarray"

        if self.target_means is None or self.target_stds is None:
            raise RuntimeError("Normalizer must be fitted before transform().")

        l, a, b = self.split_channels_lab(img)
        means, stds = self.get_mean_std(img)

        eps = 1e-8

        norm1 = ((l - means[0]) * (self.target_stds[0] / (stds[0] + eps))) + self.target_means[0]
        norm2 = ((a - means[1]) * (self.target_stds[1] / (stds[1] + eps))) + self.target_means[1]
        norm3 = ((b - means[2]) * (self.target_stds[2] / (stds[2] + eps))) + self.target_means[2]

        img_norm = self.merge(norm1, norm2, norm3)
        mask = get_tissue_mask(img)

        img_norm[~mask] = img[~mask]
        return img_norm

    @staticmethod
    def split_channels_lab(img):
        """
        Convert an RGB image to L, a, b channels (float) with conventional scaling.

        Parameters
        ----------
        img : np.ndarray
            RGB uint8 image.

        Returns
        -------
        tuple of np.ndarray
            (L, a, b) arrays as float where L is in range ~[0,100] and a/b are
            centered around 0.
        """
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img = img.astype(np.float32)

        l, a, b = cv2.split(img)

        l /= 2.55
        a -= 128.0
        b -= 128.0

        return l, a, b

    @staticmethod
    def merge(l, a, b):
        """
        Merge L, a, b channels back into an RGB uint8 image.

        Parameters
        ----------
        l : np.ndarray
            L channel in the same scale as returned by :py:meth:`split_channels_lab`.
        a : np.ndarray
            a channel offset (centered around 0).
        b : np.ndarray
            b channel offset (centered around 0).

        Returns
        -------
        np.ndarray
            RGB uint8 image obtained by converting from LAB.
        """
        l = l * 2.55
        a = a + 128.0
        b = b + 128.0

        img = cv2.merge((l, a, b))
        img = np.clip(img, 0, 255).astype(np.uint8)

        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def get_mean_std(self, img):
        """
        Compute mean and standard deviation of L, a, b channels for an image.

        Parameters
        ----------
        img : np.ndarray
            RGB uint8 image.

        Returns
        -------
        means : np.ndarray
            Array of shape (3,) with channel means (L, a, b).
        stds : np.ndarray
            Array of shape (3,) with channel standard deviations (L, a, b).
        """
        l, a, b = self.split_channels_lab(img)

        m_l, sd_l = cv2.meanStdDev(l)
        m_a, sd_a = cv2.meanStdDev(a)
        m_b, sd_b = cv2.meanStdDev(b)

        means = np.array([m_l[0][0], m_a[0][0], m_b[0][0]])
        stds = np.array([sd_l[0][0], sd_a[0][0], sd_b[0][0]])

        return means, stds

class StainingNormalizer:
    """
    Factory-like wrapper that constructs a concrete normalizer by name.

    Supported methods:
    - "macenko" : StainMatrixNormalizer with MacenkoExtractor
    - "vahadane": StainMatrixNormalizer with VahadaneExtractor
    - "reinhard": ReinhardNormalizer

    Parameters
    ----------
    method : str
        Normalization method name (case-insensitive).
    **kwargs : dict
        Method-specific keyword arguments forwarded to the extractor or normalizer.

    Raises
    ------
    ValueError
        If `method` is not recognized.
    """
    def __init__(self, method: str, **kwargs):

        method = method.lower()

        if method == "macenko":

            extractor = MacenkoExtractor(
                ang_per=kwargs.get("ang_per", 99),
                lum_thr=kwargs.get("lum_thr", 0.8),
            )

            self.normalizer = StainMatrixNormalizer(extractor)

        elif method == "vahadane":

            extractor = VahadaneExtractor(
                lum_thr=kwargs.get("lum_thr", 0.8),
                reg=kwargs.get("reg", 0.1),
                max_iter=kwargs.get("max_iter", 10),
            )

            self.normalizer = StainMatrixNormalizer(extractor)

        elif method == "reinhard":

            self.normalizer = ReinhardNormalizer(
                lum_thr=kwargs.get("lum_thr", 0.90)
            )

        else:
            raise ValueError(
                f"Unknown stain normalization method: {method}"
            )

    def fit(self, target):
        """
        Fit the underlying normalizer to a target image.

        Parameters
        ----------
        target : np.ndarray
            RGB uint8 target image.

        Returns
        -------
        StainingNormalizer
            Self with set target.
        """
        self.normalizer.fit(target.copy())
        return self

    def transform(self, img):
        """
        Apply the underlying normalizer's transform to an image.

        Parameters
        ----------
        img : np.ndarray
            RGB uint8 image to normalize.

        Returns
        -------
        np.ndarray
            Normalized RGB uint8 image.
        """
        return self.normalizer.transform(img.copy())