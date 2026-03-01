import warnings
import numpy as np
from .gmm import GaMRed_hist, get_pixel_distribution

def get_thr_image(img: np.ndarray, thr_min = 0.7*255, verbose=False):
    """
    Compute per-channel thresholds for an RGB image using the GaMRed algorithm.

    This function estimates thresholds for each color channel (Red, Green, Blue)
    of an RGB image using the Gaussian Mixture Reduction (GaMRed) method. If a
    computed threshold is lower than `thr_min`, the function falls back to Otsu's
    method for that channel. The function also returns the histogram of pixel
    values for each channel.

    Parameters
    ----------
    img : ndarray of shape (M, N, 3)
        Input RGB image with pixel values in the range [0, 255].
    thr_min : float, optional
        Minimum allowable threshold. If a GaMRed threshold is below this value,
        Otsu's method is used instead. Default is 0.7*255.
    verbose : bool, optional
        If True, prints messages when Otsu's method is used due to low thresholds.
        Default is False.

    Returns
    -------
    thr : dict
        Dictionary of thresholds for each color channel:
        - ``"R"`` : float, threshold for the Red channel
        - ``"G"`` : float, threshold for the Green channel
        - ``"B"`` : float, threshold for the Blue channel
    R : ndarray of shape (256,)
        Histogram of Red channel pixel values.
    G : ndarray of shape (256,)
        Histogram of Green channel pixel values.
    B : ndarray of shape (256,)
        Histogram of Blue channel pixel values.

    Notes
    -----
    - Uses :func:`get_pixel_distribution` to compute per-channel histograms.
    - Thresholds are initially estimated using :func:`GaMRed_hist`.
    - If a threshold is below `thr_min`, the function uses :func:`two_step_otsu`
      as a fallback for robustness.
    - `K=2` and `SW=5` are fixed parameters for the GaMRed algorithm.

    Examples
    --------
    >>> thr, R, G, B = get_thr_image(image, thr_min=180, verbose=True)
    >>> print("Thresholds:", thr)
    >>> print("Red channel histogram:", R)
    """

    x = np.arange(256)

    R, G, B = get_pixel_distribution(img)
    hist = {"R": R,
            "G": G,
            "B": B}

    thr = {"R": GaMRed_hist(x, hist["R"])[0],
           "G": GaMRed_hist(x, hist["G"])[0],
           "B": GaMRed_hist(x, hist["B"])[0]}

    for k, v in thr.items():
        if v < thr_min:
            thr[k] = two_step_otsu(hist=hist[k])
            if verbose:
                print(f"Too low threshold for {k} channel, use Otsu instead.")

    return thr, R, G, B

def two_step_otsu(hist):
    """
    Compute a threshold using a two-step Otsu algorithm.

    This function applies a hierarchical, two-step version of Otsu's method
    to determine a threshold from a histogram. The first Otsu threshold divides
    the histogram roughly in half, and the second Otsu threshold refines the
    separation within the upper segment of the histogram. This is useful for
    images with uneven lighting or bimodal intensity distributions.

    Parameters
    ----------
    hist : ndarray of shape (256,)
        Histogram of pixel intensities (counts per bin).

    Returns
    -------
    thr : float
        Computed threshold value in the range [0, 255].

    Notes
    -----
    - Relies on :func:`otsuthresh` (assumed available) for standard Otsu threshold computation.
    - The second step focuses on the upper portion of the histogram to refine
      the threshold.
    - The final threshold is scaled to the 0–255 range and rounded to the nearest integer.

    Examples
    --------
    >>> thr = two_step_otsu(hist)
    >>> print("Two-step Otsu threshold:", thr)

    """

    tmp, _ = otsuthresh(hist)
    tmp = int(tmp*255)
    tmp2, _ = otsuthresh(hist[tmp-1:])
    thr = np.round(tmp+(255-tmp)*tmp2)
    return thr

def otsuthresh(counts):
    """
    Compute Otsu's threshold and effectiveness metric for a histogram.

    This function implements Otsu's method in Python, based on the MATLAB implementation.
    It calculates the threshold that maximizes the between-class variance for a histogram
    of pixel counts, along with an effectiveness metric indicating the separation quality.

    Parameters
    ----------
    counts : array_like
        Histogram of pixel intensities (counts per bin).

    Returns
    -------
    t : float
        Normalized threshold value in the range [0, 1].
    em : float
        Effectiveness metric of the threshold (ratio of between-class variance
        to total variance). Higher values indicate better separation.

    Notes
    -----
    - Counts are converted to probabilities and cumulative sums are computed
      to calculate the between-class variance.
    - NaN values arising from division by zero are safely replaced with `-inf`.
    - Function is based on MATLAB's implementation.

    Examples
    --------
    >>> t, em = otsuthresh(hist)
    >>> print("Normalized Otsu threshold:", t)
    >>> print("Effectiveness metric:", em)

    References
    ----------

    Algorithm is described in \ :footcite:p:`Otsu`

    .. footbibliography::
    """

    counts = np.asarray(counts, dtype=np.float64).ravel()
    num_bins = counts.size

    # Probabilities
    p = counts / counts.sum()

    # Cumulative sums
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, num_bins+1))
    mu_t = mu[-1]

    # Between-class variance
    with warnings.catch_warnings():
        # Ignore invalid value encountered in divide (handled in the next lines of code)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))

    # Handle NaNs (avoid division by zero cases)
    sigma_b_squared = np.nan_to_num(sigma_b_squared, nan=-np.inf)

    maxval = sigma_b_squared.max()

    if np.isfinite(maxval) and maxval > 0:
        idx = np.mean(np.where(sigma_b_squared == maxval)[0]) + 1
        # Normalize threshold
        t = (idx - 1) / (num_bins - 1)
    else:
        t = 0.0

    # Effectiveness metric
    if np.isfinite(maxval) and maxval > 0:
        em = maxval / (np.sum(p * (np.arange(1, num_bins + 1) ** 2)) - mu_t ** 2)
    else:
        em = 0.0

    return t, em