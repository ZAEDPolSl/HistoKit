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
    approximate signal (by operation of binning).
    :param x: sample to partition
    :param y: sample to partition
    :param K: number of partitions
    :return: alpha - weights, mu - means, sigma - standard deviations
    """
    # parameters
    par1 = 0.1 # for robustness (fine for data in range 0-20)
    par2 = 10 # min number of points in signal fragment

    # initialize
    s_corr = ((x[1]-x[0])**2)/12 # sheppards correction for binned data
    K = K-1
    N = len(x)
    p_opt_idx = np.zeros((1,N))
    p_aux = np.zeros((1, N))
    opt_pals = np.zeros((K, N))

    for a in range(N):
        invec = x[a:N]
        yinvec = y[a:N]
        if np.sum(yinvec)<=par2:
            p_opt_idx[a] = np.inf
        else:
            wwec = yinvec/(np.sum(yinvec))
            var_bin = np.sum()


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