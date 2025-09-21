import numpy as np
from scipy.stats import norm

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
    pass

def EM_iter_hist(x, y, alpha, mu, sig, SW):
    pass

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


    return alpha.reshape(1, -1), mu.reshape(1, -1), sigma.reshape(1, -1)

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
    pass

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

    # reshape to ndarray of size 1x256
    R = R.reshape(1, -1)
    G = G.reshape(1, -1)
    B = B.reshape(1, -1)
    return R, G, B


def otsu_thresh():
    pass