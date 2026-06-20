import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from sklearn.decomposition import sparse_encode
from src.histokit.stain_normalisation.exceptions import StainNormalizationError


def rgb2od(img, eps = 1e-6):
    """
    Convert an RGB image to optical density (OD) space.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image with dtype ``np.uint8`` and shape ``(H, W, 3)``.
    eps : float, optional
        Constant to avoid ``log(0)``. Default is ``1e-6``.

    Returns
    -------
    numpy.ndarray
        Optical density image with dtype ``np.float32`` and shape ``(H, W, 3)``.
    """
    img[(img == 0)] = 1
    img = img.astype(np.float32)
    od = np.maximum(-np.log(img / 255), eps)
    return od

def od2rgb(od):
    """
    Convert optical density (OD) image to RGB.

    Parameters
    ----------
    od : numpy.ndarray
        Optical density image (float). Expected to be non-negative.

    Returns
    -------
    numpy.ndarray
        RGB image with dtype ``np.uint8``.

    Raises
    ------
    StainNormalizationError
        If any value in ``od`` is negative.
    """
    if od.min() >= 0: StainNormalizationError("OD values must be non-negative.")
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


def normalize_matrix(matrix):
    """
    Normalize each row of a matrix to unit L2 length.

    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix of shape ``(N, M)`` where each row will be normalized.

    Returns
    -------
    numpy.ndarray
        Row-normalized matrix.

    Raises
    ------
    ValueError
        If any row has zero L2 norm and therefore cannot be normalized.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Cannot normalize rows with zero L2 norm.")

    return matrix / norms

def get_concentrations(img, stain_matrix, regularizer=0.01):
    """
    Estimate concentration coefficients for each pixel given a stain matrix.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image with dtype ``np.uint8`` and shape ``(H, W, 3)``.
    stain_matrix : numpy.ndarray
        Dictionary / stain matrix used as the sparse encoding dictionary.
        Shape is typically ``(n_stains, 3)`` (or compatible with ``sparse_encode``).
    regularizer : float, optional
        Regularization parameter passed as ``alpha`` to ``sklearn.decomposition.sparse_encode``.
        Default is ``0.01``.

    Returns
    -------
    numpy.ndarray
        Concentration matrix with shape ``(H*W, n_stains)`` containing the estimated
        per-pixel concentrations.

    Notes
    -----
    The image is converted to OD space and reshaped to ``(H*W, 3)`` before sparse encoding.
    """
    od = rgb2od(img).reshape((-1, 3))
    return sparse_encode(X=od, dictionary=stain_matrix, algorithm="lasso_lars", alpha=regularizer, positive=True)

def is_rgb_uint8(img):
    """
    Check whether an array is an RGB image with dtype ``uint8``.

    Parameters
    ----------
    img : numpy.ndarray
        Input array.

    Returns
    -------
    bool
        ``True`` if ``img`` has dtype ``np.uint8`` and ``ndim == 3`` (i.e. RGB), otherwise ``False``.
    """
    if img.dtype == np.uint8 and img.ndim == 3:
        return True
    return False

def get_tissue_mask(
    img,
    bg_percentile=95,
    margin=0.03,
    min_sat=0.05,
    min_chroma=0.02,
    min_size=50
):
    """
        Compute a tissue mask combining luminosity, saturation and chroma.

        Parameters
        ----------
        img : numpy.ndarray
            RGB image with dtype ``np.uint8`` and shape ``(H, W, 3)``.
        bg_percentile : float, optional
            Percentile of the L channel used to estimate background lightness (0-100).
            Default is ``95``.
        margin : float, optional
            Subtracted from the background percentile to get the luminosity threshold.
            Default is ``0.03``.
        min_sat : float, optional
            Minimum saturation (0-1) to be considered tissue. Default is ``0.05``.
        min_chroma : float, optional
            Minimum chroma (0-1) to be considered tissue. Default is ``0.02``.
        min_size : int, optional
            Minimum connected component size to keep (pixels). Default is ``50``.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape ``(H, W)`` where ``True`` indicates tissue.

        Raises
        ------
        StainNormalizationError
            If input is not RGB ``uint8`` or if the computed mask is empty.
    """
    if not is_rgb_uint8(img):
        raise StainNormalizationError("Input image must be RGB uint8.")

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    L = img_lab[:, :, 0] / 255.0
    a = img_lab[:, :, 1].astype(np.float32) - 128
    b = img_lab[:, :, 2].astype(np.float32) - 128

    chroma = np.sqrt(a**2 + b**2) / 181.0

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S = img_hsv[:, :, 1] / 255.0

    lum_thr = np.percentile(L, bg_percentile) - margin
    lum_thr = np.clip(lum_thr, 0.75, 0.98)

    mask = (L < lum_thr) | (S > min_sat) | (chroma > min_chroma)
    mask = remove_small_objects(mask, min_size=min_size)

    if mask.sum() == 0:
        raise StainNormalizationError("Empty mask computed.")

    return mask

