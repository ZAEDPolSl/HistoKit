import cv2
import random
import numpy as np
from .base import Transform

class GaussianBlur(Transform):
    """
    Apply Gaussian blur to an image.

    The blur strength is controlled by the standard deviation `sigma`,
    which is randomly sampled from `sigma_range` each time the transform
    is applied. Size of the Gaussian kernel is automatically determined 
    based on the sampled `sigma`. Uniform random sampling is used for the `sigma` value.

    Parameters
    ----------
    sigma_range : tuple of float, default: (0.3, 2.0)
        Range from which the Gaussian sigma value is sampled.
    prob : float, (default: 1.0)
        Probability of applying the augmentation.
    """
    def __init__(self, sigma_range=(0.3, 2.0), prob=1.0):
        """
        Initialize the Gaussian blur augmentation.

        Parameters
        ----------
        sigma_range : tuple of float, default: (0.3, 2.0)
            Minimum and maximum sigma values used for Gaussian blur.
        prob : float, (default: 1.0)
            Probability of applying the augmentation.
        """
        super().__init__(prob)
        self.sigma_range = sigma_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to the input image.

        Parameters
        ----------
        img : np.ndarray
            Input image represented as a NumPy array.

        Returns
        -------
        np.ndarray
            Blurred image.
        """
        sigma = random.uniform(*self.sigma_range)
        return cv2.GaussianBlur(img, (0, 0), sigma)


class MedianBlur(Transform):
    """
    Apply median blur to an image.

    The kernel size is randomly sampled from `size_range`. If the sampled
    size is even, it is increased by one (OpenCV median blur requires
    an odd kernel size). Uniform random sampling is used for the kernel size.

    Parameters
    ----------
    size_range : tuple of int, default: (3, 7)
        Range from which the median blur kernel size is sampled.
    prob : float, (default: 1.0)
        Probability of applying the transformation.
    """
    def __init__(self, size_range=(3, 7), prob=1.0):
        """
        Initialize the median blur transform.

        Parameters
        ----------
        size_range : tuple of int, default: (3, 7)
            Minimum and maximum kernel size values.
        prob : float, (default: 1.0)
            Probability of applying the transform.
        """
        super().__init__(prob)
        self.size_range = size_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply median blur to the input image.

        Parameters
        ----------
        img : np.ndarray
            Input image represented as a NumPy array.

        Returns
        -------
        np.ndarray
            Image after applying median blur.
        """
        size = random.randint(*self.size_range)

        if size % 2 == 0:
            size += 1

        return cv2.medianBlur(img, size)


class MotionBlur(Transform):
    """
    Apply motion blur to an image.

    A linear blur kernel is created with a randomly sampled degree and then
    rotated by a randomly sampled angle. The resulting kernel is applied to
    the image using 2D convolution. Uniform random sampling is used for both 
    the degree and angle of the motion blur.

    Parameters
    ----------
    degree_range : tuple of int, default: (5, 15)
        Range from which the motion blur kernel size is sampled.
        Higher values produce stronger blur.
    angle_range : tuple of float, default: (0, 360)
        Range of rotation angles in degrees used for the motion blur direction.
    prob : float, default: 1.0
        Probability of applying the transformation.
    """
    def __init__(
        self,
        degree_range=(5, 15),
        angle_range=(0, 360),
        prob=1.0):

        super().__init__(prob)
        self.degree_range = degree_range
        self.angle_range = angle_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply motion blur to the input image.

        Parameters
        ----------
        degree_range : tuple of int, optional
            Minimum and maximum motion blur kernel sizes.
        angle_range : tuple of float, optional
            Minimum and maximum motion blur angles in degrees.
        prob : float, optional
            Probability of applying the transform.
        """
        degree = random.randint(*self.degree_range)
        angle = random.uniform(*self.angle_range)

        kernel = np.zeros((degree, degree), dtype=np.float32)
        kernel[(degree - 1) // 2, :] = 1.0
        kernel /= degree

        M = cv2.getRotationMatrix2D(
            (degree / 2 - 0.5, degree / 2 - 0.5),
            angle,
            1.0,
        )

        kernel = cv2.warpAffine(
            kernel,
            M,
            (degree, degree),
        )

        return cv2.filter2D(img, -1, kernel)


class JPEGCompression(Transform):
    """
    Perform JPEG compression.

    The image is compressed to JPEG format using a randomly sampled quality value
    The Lower quality values produce stronger
    compression artifacts.

    Parameters
    ----------
    quality_range : tuple of int, default: (10, 50)
        Range from which the JPEG quality value is sampled.
        Values should usually be between 0 and 100.
    prob : float, default: 1.0
        Probability of applying the transformation.
    """
    def __init__(
        self,
        quality_range=(10, 50),
        prob=1.0,
    ):
        super().__init__(prob)
        self.quality_range = quality_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply JPEG compression to the input image.

        Parameters
        ----------
        quality_range : tuple of int, default: (10, 50)
            Range from which the JPEG quality value is sampled.
        prob : float, default: 1.0
            Probability of applying the transform.
        """
        quality = random.randint(*self.quality_range)

        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

        if not success:
            return img

        decoded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        return cv2.cvtColor(
            decoded,
            cv2.COLOR_BGR2RGB,
        )
    

class SaltAndPepper(Transform):
    """Randomly add salt-and-pepper noise to an image.

    This transform randomly replaces a fraction of image pixels with either
    white values, called salt, or black values, called pepper.

    Parameters
    ----------
    amount_range : tuple[float, float], default=(0.01, 0.05)
        Range used to sample the fraction of pixels affected by noise.
        The sampled value is multiplied by the image area ``H * W`` to compute
        the number of noisy pixels.
    prob : float, default=1.0
        Probability of applying the transform.

    Notes
    -----
    The input image is expected to have pixel values in ``[0, 255]``.
    The returned image has the same shape and dtype as the input image.
    """
    def __init__(self, amount_range=(0.01, 0.05), prob=1.0):
        super().__init__(prob)
        self.amount_range = amount_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply salt-and-pepper noise to an image.

        Parameters
        ----------
        img : np.ndarray
            Input image with shape ``(H, W)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Image with randomly selected pixels replaced by 0 or 255. The shape
            and dtype are the same as in ``img``.
        """
        arr = img.copy()
        amount = random.uniform(*self.amount_range)

        h, w = arr.shape[:2]
        num_pixels = int(amount * h * w)

        num_salt = num_pixels // 2
        num_pepper = num_pixels - num_salt

        ys = np.random.randint(0, h, num_salt)
        xs = np.random.randint(0, w, num_salt)
        arr[ys, xs] = 255

        ys = np.random.randint(0, h, num_pepper)
        xs = np.random.randint(0, w, num_pepper)
        arr[ys, xs] = 0

        return arr


class GaussianNoise(Transform):
    """Add random Gaussian noise to an image.

    This transform samples a Gaussian noise distribution with random mean and
    standard deviation, adds the noise to the input image, and clips the result
    to the valid image intensity range ``[0, 255]``.

    Parameters
    ----------
    mean_range : tuple[float, float], default=(0.0, 6.0)
        Range used to sample the Gaussian noise mean. The value is sampled as
        ``random.uniform(*mean_range)``.
    std_range : tuple[float, float], default=(10.0, 30.0)
        Range used to sample the Gaussian noise standard deviation. The value is
        sampled as ``random.uniform(*std_range)``.
    prob : float, default=1.0
        Probability of applying the transform.

    Notes
    -----
    The input image is expected to have pixel values in ``[0, 255]``.
    The returned image has dtype ``np.uint8``.
    """
    def __init__(self, mean_range=(0.0, 6.0), std_range=(10.0, 30.0), prob=1.0):
        super().__init__(prob)
        self.mean_range = mean_range
        self.std_range = std_range

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise to an image.

        Parameters
        ----------
        img : np.ndarray
            Input image with shape ``(H, W)`` or ``(H, W, C)`` and pixel values
            in ``[0, 255]``.

        Returns
        -------
        np.ndarray
            Image with added Gaussian noise. The output has the same shape as
            ``img`` and dtype ``np.uint8``.
        """
        mean = random.uniform(*self.mean_range)
        std = random.uniform(*self.std_range)

        arr = img.astype(np.float32)
        noise = np.random.normal(mean, std, arr.shape)

        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr