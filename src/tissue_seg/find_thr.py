import warnings
import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, color, filters

def GaMRed_hist(x, y, K, draw, SW):
    """
    Estimating noise components using Gaussian mixture model.
    :param x: binned data
    :param y: binned data
    :param K: number of gaussian components
    :param draw: draw plots or not
    :param SW:
    :return:

    Matalab implementation: Michal Marczyk (Michal.Marczyk@polsl.pl)
    """
    ind = np.argsort(x)
    x = np.sort(x)
    y = y[ind]
    N = np.sum(y) #nb of measurments
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
    Expectation maximisation algorithm iteration.
    :param x: binned data (bins)
    :param y: binned data (counts)
    :param alpha: GMM components' weights
    :param mu: GMM components' means
    :param sig: GMM components' standard deviations
    :param SW:
    :return: alpha - updated weights, mu - updated means, sigma - updated standard deviations, logL - loglikelihood
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
    Compute initial conditions for GMM by using dynamic programming for
    approximate signal (by operation of binning). There are small numerical
    differences (around 2.84217094e-14) due to different accuracy of calculations between Matlab and Python.
    :param x: sample to partition
    :param y: sample to partition
    :param K: number of partitions
    :return: alpha - weights, mu - means, sigma - standard deviations
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
    Get values from normal distribution, there are
    small (10^(-15)) differences between Matlab and scipy implementations.
    :param x: vector of arguments
    :param mu: mean of normal distribution
    :param sigma: standard deviation
    :return: values from normal distribution
    """
    y = norm.pdf(x, mu, sigma)
    return y.reshape(1, -1)

def find_thr(data, alpha, mi, sigma, idx, draw):
    """
    Find threshold
    :param data: data
    :param alpha: GMM component weights
    :param mi: GMM component means
    :param sigma: GMM component standard deviations
    :param idx: index for informative/non-informative components
    :param draw: draw plot or not
    :return: thr - threshold value
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
    Get distribution of pixel color values per color channel.
    :param img: ndarray with RGB image
    :return: R, G, B - counts of pixel color values for each channel
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
    Get thresholds for RGB image calculated using GaMRed algorithm.
    When threshold is lower than thr_min threshold is calculated using
    Otsu method.
    :param img: numpy ndarray with RGB image
    :param thr_min: if threshold calculated with GaMRed is lower than thr_min, use Otsu instead.
    :param verbose: print communicates or not
    :return: thr - threshold value for each color channel (dictionary) R, G, B - arrays with pixel counts.
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
    Two-step Otsu algorithm implementation.
    :param hist: array with pixel counts
    :return: thr - threshold value
    """
    tmp, _ = otsuthresh(hist)
    tmp = int(tmp*255)
    tmp2, _ = otsuthresh(hist[tmp-1:])
    thr = np.round(tmp+(255-tmp)*tmp2)
    return thr

def otsuthresh(counts):
    """
    Python implementation of Otsu's method, based on matlab's implementation.
    :param counts: array with pixel counts
    :return: t - threshold, em - effectiveness metric
    """
    counts = np.asarray(counts, dtype=np.float128).ravel()
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

