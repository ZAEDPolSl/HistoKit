import warnings
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def GaMRed_hist(x, y, K, draw, SW):
    """
    Estimate noise components using a Gaussian Mixture Model (GMM) fitted to histogram data.

    This function fits a Gaussian mixture model to histogrammed data using the
    Expectation-Maximization (EM) algorithm. It computes the Bayesian Information
    Criterion (BIC) for model evaluation and determines a threshold separating
    the mixture components, if applicable. The function returns the threshold,
    BIC value, and a dictionary of fitted model statistics.

    Parameters
    ----------
    x : ndarray of shape (M,)
        Bin centers of the histogram (can be unsorted; will be sorted internally).
    y : ndarray of shape (M,)
        Bin counts (frequency of observations for each bin).
    K : int
        Number of Gaussian components to fit in the mixture model.
    draw : bool
        Whether to print diagnostic information or draw plots.
    SW : object
        Settings structure or dictionary controlling the EM algorithm behavior
        (e.g., convergence criteria, maximum iterations). The exact format
        depends on the helper functions (e.g., :func:`EM_iter_hist`).

    Returns
    -------
    thr : float
        Threshold value separating mixture components along the x-axis.
        If no threshold can be determined, returns ``np.nan``.
    bic : float
        Bayesian Information Criterion (BIC) value for the fitted model.
    stats : dict
        Dictionary containing model statistics and parameters:
        - ``"thr"`` : float, threshold value
        - ``"alpha"`` : ndarray, mixture component weights
        - ``"mu"`` : ndarray, component means
        - ``"K"`` : int, number of fitted components
        - ``"sigma"`` : ndarray, component standard deviations
        - ``"logL"`` : float, log-likelihood of the fitted model

    Raises
    ------
    ValueError
        If the EM algorithm fails (e.g., returns invalid log-likelihood leading
        to infinite or zero BIC), prompting a rerun with different initialization.

    Notes
    -----
    - For ``K == 1``, no separation is possible, and the threshold is set
      below the minimum x value.
    - For ``K == 2``, the threshold is computed directly between the two Gaussian
      components.
    - For ``K > 2``, components are grouped into two clusters using
      :class:`sklearn.cluster.KMeans` before determining the separating threshold.

    References
    ----------
    MATLAB implementation : Michal Marczyk (Michal.Marczyk@polsl.pl)
    Algorithm is described in \ :footcite:p:`Marczyk2018GaMRed`

    Examples
    --------
    >>> thr_back = {"R": 0.6, "G": 0.6, "B": 0.6}
    >>> thr, bic, stats = GaMRed_hist(x, y, K=2, draw=False, SW=my_settings)
    >>> print(f"Threshold: {thr:.3f}, BIC: {bic:.2f}")
    >>> print("Component means:", stats["mu"])
    """
    ind = np.argsort(x)
    x = np.sort(x)
    y = y[ind]
    N = np.sum(y) #nb of measurements
    bic = np.inf

    # remove no signal at the beginning or the end
    y[0] = 0
    y[-1] = 0

    # initial conditions
    if K == 1:
        alpha_init = 1
        mi_init = np.mean(x)
        sigma_init = np.std(x)
    else:
        alpha_init, mi_init, sigma_init = gmm_init_dp_hist(x, y, K)

    if draw:
        print("Starting values")
        print(f"alpha_init: {alpha_init}, mi_init: {mi_init}, sigma_init: {sigma_init}")

    while bic == np.inf or bic == 0:
        # EM algorithm
        alpha, mi, sigma, logL = EM_iter_hist(x, y, alpha_init, mi_init, sigma_init, SW)

        # calculating BIC
        bic = -2*logL + (3*K-1)*np.log(N)
        if bic == np.inf or bic == 0:
            raise ValueError("EM crash. Repeat calculations.")

    ind = np.argsort(mi)
    mi  = np.sort(mi)
    alpha = alpha[ind]
    sigma = sigma[ind]

    if draw:
        print("Final values")
        print(f"alpha: {alpha}, mi: {mi}, sigma: {sigma}")

    # find threshold between components
    if K == 1:
        thr = np.min(x) - 1e-10
    elif K == 2:
        thr = find_thr(x, alpha, mi, sigma, np.array([0, 1]), draw)
    else:
        temp = np.column_stack((alpha, mi, sigma))
        kmeans = KMeans(n_clusters=2, n_init=50, random_state=0).fit(temp)
        idx = kmeans.labels_
        thr = find_thr(x, alpha, mi, sigma,idx-1, draw)

    try:
        thr
    except NameError:
        thr = np.nan

    stats = {
        "thr": thr,
        "alpha": alpha,
        "mu": mi,
        "K": K,
        "sigma": sigma,
        "logL": logL,
    }

    return thr, bic, stats

def EM_iter_hist(x, y, alpha, mu, sig, SW):
    """
    Perform one iteration of the Expectation-Maximization (EM) algorithm for a Gaussian Mixture Model (GMM) on binned data.

    The function iteratively updates the mixture weights, means, and standard deviations
    of a GMM fitted to histogram data using the EM procedure until convergence or
    a maximum number of iterations is reached. It returns the updated parameters
    and the log-likelihood of the fitted model.

    Parameters
    ----------
    x : ndarray of shape (N,)
        Bin centers of the histogram.
    y : ndarray of shape (N,)
        Counts or frequencies for each bin.
    alpha : ndarray of shape (K,)
        Initial weights of the Gaussian components (sum to 1).
    mu : ndarray of shape (K,)
        Initial means of the Gaussian components.
    sig : ndarray of shape (K,)
        Initial standard deviations of the Gaussian components.
    SW : float
        Minimum allowed standard deviation (safeguard against too small variances).

    Returns
    -------
    alpha : ndarray of shape (K,)
        Updated weights of the Gaussian components.
    mu : ndarray of shape (K,)
        Updated means of the Gaussian components, sorted in ascending order.
    sig : ndarray of shape (K,)
        Updated standard deviations of the Gaussian components, sorted according to `mu`.
    logL : float
        Log-likelihood of the histogram data under the fitted GMM.

    Notes
    -----
    - The algorithm stops when the change in parameters is below a threshold (1e-6)
      or after 10,000 iterations.

    Examples
    --------
    >>> alpha, mu, sig, logL = EM_iter_hist(x, y, alpha_init, mu_init, sig_init, SW=0.01)
    >>> print("Updated means:", mu)
    >>> print("Log-likelihood:", logL)
    """

    N = len(y)
    n = np.sum(y)
    sig2 = np.maximum(sig ** 2, SW ** 2)
    change = np.inf
    count = 1
    SW = SW ** 2
    eps_change = 1e-6
    KS = len(alpha)

    while change > eps_change and count < 10000:
        old_alpha = alpha.copy()
        old_sig2 = sig2.copy()

        f = np.zeros((KS, N))
        sig = np.sqrt(sig2)

        for a in range(KS):
            f[a, :] = norm_pdf(x, mu[a], sig[a])

        px = alpha @ f
        px[np.isnan(px) | (px == 0)] = 5e-324

        for a in range(KS):
            pk = ((alpha[a] * f[a, :]) * y) / px
            denom = np.sum(pk)
            mu[a] = (pk @ x) / denom
            sig2num = np.sum(pk @ ((x - mu[a]) ** 2))
            sig2[a] = np.maximum(SW, sig2num / denom)
            alpha[a] = denom / n

        change = np.sum(np.abs(alpha - old_alpha)) + np.sum(np.abs(sig2 - old_sig2) / sig2) / len(alpha)
        count += 1

    # return results
    logL = np.sum(np.log(px) * y)
    mu_est = np.sort(mu)
    ind = np.argsort(mu)
    sig_est = np.sqrt(sig2[ind])
    pp_est = alpha[ind]

    return pp_est, mu_est, sig_est, logL

def gmm_init_dp_hist(x, y, K):
    """
    Compute initial parameters for a Gaussian Mixture Model using dynamic programming on binned data.

    This function estimates starting values for the mixture weights, means, and standard deviations
    of a GMM by partitioning the histogram (binned data) into K segments.

    Parameters
    ----------
    x : ndarray of shape (N,)
        Bin centers of the histogram.
    y : ndarray of shape (N,)
        Counts (frequencies) of each bin.
    K : int
        Number of Gaussian components (partitions) to initialize.

    Returns
    -------
    alpha : ndarray of shape (K,)
        Initial weights of the Gaussian components, proportional to the sum of counts in each partition.
    mu : ndarray of shape (K,)
        Initial means of the Gaussian components, computed as weighted averages of bin centers within each partition.
    sigma : ndarray of shape (K,)
        Initial standard deviations of the Gaussian components, corrected for binned data using Sheppard's correction.

    Notes
    -----
    - Sheppard's correction is applied to account for variance underestimation due to binning:
      `s_corr = ((x[1] - x[0]) ** 2) / 12`.

    Examples
    --------
    >>> alpha_init, mu_init, sigma_init = gmm_init_dp_hist(x, y, K=3)
    >>> print("Initial weights:", alpha_init)
    >>> print("Initial means:", mu_init)
    >>> print("Initial standard deviations:", sigma_init)
    """
    # parameters
    par1 = 0.1  # for robustness (fine for data in range 0-20)
    par2 = 10  # min number of points in signal fragment

    # initialize
    s_corr = ((x[1] - x[0]) ** 2) / 12  # sheppards correction for binned data
    K = K - 1
    N = len(x)
    p_opt_idx = np.zeros(N)
    p_aux = np.zeros(N)
    opt_pals = np.zeros((K, N))

    for a in range(N):
        invec = x[a:N]
        yinvec = y[a:N]
        if np.sum(yinvec) <= par2:
            p_opt_idx[a] = np.inf
        else:
            wwec = yinvec / (np.sum(yinvec))
            var_bin = np.sum(((invec - np.sum(invec * wwec)) ** 2) * wwec)
            if var_bin > s_corr:
                p_opt_idx[a] = (par1 + np.sqrt(var_bin - s_corr)) / (np.max(invec) - np.min(invec))
            else:
                p_opt_idx[a] = np.inf

    # diff p_opt_idx => e-17

    # aux mx
    aux_mx = np.zeros((N, N))
    for a in range(N - 1):
        for b in range(a + 1, N):
            invec = x[a:b]
            yinvec = y[a:b]
            if np.sum(yinvec) <= par2:
                aux_mx[a, b] = np.inf
            else:
                wwec = yinvec / (np.sum(yinvec))
                var_bin = np.sum(((invec - np.sum(invec * wwec)) ** 2) * wwec)
                if var_bin > s_corr:
                    aux_mx[a, b] = (par1 + np.sqrt(var_bin - s_corr)) / (np.max(invec) - np.min(invec))
                else:
                    aux_mx[a, b] = np.inf

    # iterate
    for kster in range(K):
        # kster
        for a in range(N - kster - 1):
            for b in range(a + 1, N - kster):
                p_aux[b] = aux_mx[a, b] + p_opt_idx[b]
            mm = np.min(p_aux[a + 1:N - kster])
            ix = np.argmin(p_aux[a + 1:N - kster])
            p_opt_idx[a] = mm  # e-16
            opt_pals[kster, a] = a + ix + 1

    # restore optimal decisions
    opt_part = np.zeros(K)
    opt_part[0] = opt_pals[K-1, 0]
    for kster in range(K - 2, -1, -1):
        opt_part[K - kster -1] = opt_pals[kster, int(opt_part[K - kster - 2])]

    # find initial conditions
    opt_part = np.concatenate(([0], opt_part, [N]))
    alpha = np.zeros(K + 1)
    mu = np.zeros(K + 1)
    sigma = np.zeros(K + 1)

    for a in range(K + 1):
        invec = x[int(opt_part[a]):int(opt_part[a + 1])]
        yinvec = y[int(opt_part[a]):int(opt_part[a + 1])]
        wwec = yinvec / (np.sum(yinvec))
        alpha[a] = np.sum(yinvec) / np.sum(y)
        mu[a] = np.sum(invec * wwec)
        sigma[a] = np.sqrt(np.sum(((invec - np.sum(invec * wwec)) ** 2) * wwec) - s_corr)


    return alpha, mu, sigma

def norm_pdf(x, mu, sigma):
    """
    Evaluate the probability density function (PDF) of a normal distribution.

    Computes the PDF values of a normal (Gaussian) distribution with specified
    mean and standard deviation at the given input points.

    Parameters
    ----------
    x : array_like
        Input values where the PDF is evaluated.
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution.

    Returns
    -------
    y : ndarray of shape (1, N)
        PDF values evaluated at `x`, returned as a 2D array with one row.

    Examples
    --------
    >>> y = norm_pdf(np.array([0, 1, 2]), mu=0, sigma=1)
    >>> print(y)
    """
    y = norm.pdf(x, mu, sigma)
    return y.reshape(1, -1)

def find_thr(data, alpha, mi, sigma, idx, draw):
    """
    Determine the threshold separating informative and non-informative GMM components.

    The function evaluates the Gaussian Mixture Model (GMM) over a finely spaced grid,
    computes the combined densities of informative and non-informative components,
    and identifies the threshold where the absolute difference between these densities
    is minimized. Optionally, plots the component densities.

    Parameters
    ----------
    data : ndarray
        Input data points used to determine the threshold range.
    alpha : ndarray of shape (K,)
        Weights of the GMM components.
    mi : ndarray of shape (K,)
        Means of the GMM components.
    sigma : ndarray of shape (K,)
        Standard deviations of the GMM components.
    idx : array_like of shape (K,)
        Boolean index indicating which components are considered informative (True)
        and non-informative (False).
    draw : bool
        If True, plots the component densities, their difference, and the threshold.

    Returns
    -------
    thr : float
        Threshold value along the data axis that best separates informative from
        non-informative components.

    Notes
    -----
    - The function uses a high-resolution grid (`1e7` points) for precise threshold determination.
    - If no valid index is found within the constraints, a ValueError is raised.

    Examples
    --------
    >>> idx = np.array([False, True])
    >>> thr = find_thr(data, alpha, mu, sigma, idx, draw=True)
    >>> print("Threshold value:", thr)
    """

    idx = idx.astype(bool)

    # generate data with better precision
    K = len(mi)
    f_temp=np.zeros((int(1e7), K))
    x_temp=np.linspace(np.min(data), np.max(data), int(1e7))

    for k in range(K):
        f_temp[:, k] = alpha[k]*norm_pdf(x_temp, mi[k], sigma[k])

    # find GMM for informative and non-informative components
    f1 = np.sum(f_temp[:, ~idx], axis=1)
    f2 = np.sum(f_temp[:, idx], axis=1)

    # calculate difference of f1 and f2 and find its global minimum
    f_diff = np.abs(f1 - f2)
    ind1 = np.argmax(f1)
    ind2 = np.argmax(f2)
    ind = np.argsort(f_diff)

    ind = ind[(ind<ind1) & (ind>ind2)]

    if len(ind) == 0:
        ind = np.argsort(f_diff)
        a=0
        thr_ind = ind[a]

        while thr_ind<ind1 or thr_ind>ind2:
            a+=1
            if a>=len(ind):
                raise ValueError("Missing index")
            thr_ind = ind[a]
    else:
        thr_ind = ind[0]

    thr = x_temp[thr_ind]

    if draw:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].plot(x_temp, f1, 'g', linewidth=2, label='f1')
        axes[0].plot(x_temp, f2, 'r', linewidth=2, label='f2')
        axes[0].set_xlabel("Variable")
        axes[0].set_ylabel("Model", fontsize=14)
        axes[0].legend()

        axes[1].plot(x_temp, f_diff, 'r', linewidth=2, label='f_diff')
        axes[1].set_xlabel("Variable")
        axes[1].set_ylabel("Models difference", fontsize=14)
        axes[1].set_title(f"Threshold: {thr:.4f}")

        plt.tight_layout()
        plt.show()

    return thr

def get_pixel_distribution(img):
    """
    Compute the pixel value distribution for each RGB channel of an image.

    This function calculates histograms of pixel intensities (0–255) separately
    for the Red, Green, and Blue channels of an input RGB image. Counts for
    near-white pixels (values 254 and 255) are set to zero to remove artificial
    or background artifacts.

    Parameters
    ----------
    img : ndarray of shape (M, N, 3)
        Input RGB image as a NumPy array with values in the range [0, 255].

    Returns
    -------
    R : ndarray of shape (256,)
        Histogram of Red channel pixel intensities (with counts for 254 and 255 set to zero).
    G : ndarray of shape (256,)
        Histogram of Green channel pixel intensities (with counts for 254 and 255 set to zero).
    B : ndarray of shape (256,)
        Histogram of Blue channel pixel intensities (with counts for 254 and 255 set to zero).

    Examples
    --------
    >>> R, G, B = get_pixel_distribution(image)
    >>> print("Red channel counts:", R)
    >>> print("Green channel counts:", G)
    >>> print("Blue channel counts:", B)
    """

    # get distribution of pixel values per color channel
    R = img[:, :, 0].ravel()
    G = img[:, :, 1].ravel()
    B = img[:, :, 2].ravel()

    bins = np.arange(-0.5, 256.5, 1)

    R, _ = np.histogram(R, bins=bins)
    B, _ = np.histogram(B, bins=bins)
    G, _ = np.histogram(G, bins=bins)

    # remove counts from artificial white pixels
    R[254:] = 0
    G[254:] = 0
    B[254:] = 0

    return R, G, B


def get_thr_image(img, thr_min = 0.7*255, verbose=False):
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
    K = 2
    SW = 5
    draw = False

    R, G, B = get_pixel_distribution(img)
    hist = {"R": R,
            "G": G,
            "B": B}

    thr = {"R": GaMRed_hist(x, hist["R"], K, draw, SW)[0],
           "G": GaMRed_hist(x, hist["G"], K, draw, SW)[0],
           "B": GaMRed_hist(x, hist["B"], K, draw, SW)[0]}

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

