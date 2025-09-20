import numpy as np
import pytest
from scipy.io import loadmat
from PIL import Image
from numpy.testing import assert_array_equal, assert_allclose
from src.tissue_seg.find_thr import get_pixel_distribution, norm_pdf


@pytest.mark.parametrize("image_path, mat_file", [
    ("../../test_data/test_find_thr/region_1.tif", "../../test_data/test_find_thr/pixel_distribution_1.mat"),
    ("../../test_data/test_find_thr/region_2.tif", "../../test_data/test_find_thr/pixel_distribution_2.mat"),
    ("../../test_data/test_find_thr/region_3.tif", "../../test_data/test_find_thr/pixel_distribution_3.mat")
])
def test_find_pixel_distribution(image_path, mat_file):
    img = Image.open(image_path)
    img_np = np.array(img)
    R, G, B = get_pixel_distribution(img_np)
    mat_res = loadmat(mat_file)

    assert_array_equal(R, mat_res["R"])
    assert_array_equal(G, mat_res["G"])
    assert_array_equal(B, mat_res["B"])


@pytest.mark.parametrize("image_path, mat_file", [
    ("../../test_data/test_find_thr/region_1.tif", "../../test_data/test_find_thr/GaMRed_hist_1.mat"),
    ("../../test_data/test_find_thr/region_2.tif", "../../test_data/test_find_thr/GaMRed_hist_2.mat"),
    ("../../test_data/test_find_thr/region_3.tif", "../../test_data/test_find_thr/GaMRed_hist_3.mat")
])
def test_GaMRed_hist(image_path, mat_file):
    img = Image.open(image_path)
    img_np = np.array(img)




def test_EM_iter_hist(self):
    self.assertEqual(True, False)

@pytest.mark.parametrize("channel, K, mat_file", [
    ("R",2, "../../test_data/test_find_thr/gmm_init_dp_hist_1.mat"),
    ("G",3, "../../test_data/test_find_thr/gmm_init_dp_hist_2.mat"),
    ("B",10, "../../test_data/test_find_thr/gmm_init_dp_hist_3.mat")
])
def test_gmm_init_dp_hist(channel, K, mat_file):
    img = Image.open("../../test_data/test_find_thr/region_3.tif")
    img_np = np.array(img)
    distribution = get_pixel_distribution(img_np)
    channels = {"R": distribution[0],
                "G": distribution[1],
                "B": distribution[2]}

    x = np.arange(256).reshape(1, -1)
    y = channels[channel]

    # parameters
    par1 = 0.1  # for robustness (fine for data in range 0-20)
    par2 = 10  # min number of points in signal fragment

    # initialize
    s_corr = ((x[1] - x[0]) ** 2) / 12  # sheppards correction for binned data
    K = K - 1
    N = len(x)
    p_opt_idx = np.zeros((1, N))
    p_aux = np.zeros((1, N))
    opt_pals = np.zeros((K, N))

    for a in range(N):
        invec = x[a:N]
        yinvec = y[a:N]
        if np.sum(yinvec) <= par2:
            p_opt_idx[a] = np.inf
        else:
            wwec = yinvec / (np.sum(yinvec))
            var_bin = np.sum(((invec-np.sum(invec*wwec))**2)*wwec)
            if var_bin > s_corr:
                p_opt_idx[a] = (par1+np.sqrt(var_bin-s_corr))/(np.max(invec)-np.min(invec))
            else:
                p_opt_idx[a] = np.inf

    # aux mx
    aux_mx = np.zeros((N, N))
    for a in range(N-1):
        for b in range(a+1, N):
            invec = x[a:b-1]
            yinvec = y[a:b-1]
            if np.sum(yinvec)<=par2:
                aux_mx[a, b] = np.inf
            else:
                wwec = yinvec/(np.sum(yinvec))
                var_bin = np.sum(((invec-np.sum(invec*wwec))**2)*wwec)
                if var_bin > s_corr:
                    aux_mx[a, b] = (par1+np.sqrt(var_bin-s_corr))/(np.max(invec)-np.min(invec))
                else:
                    aux_mx[a, b] = np.inf

    # iterate
    for kster in range(K):
        # kster
        for a in range(N-kster):
            for b in range(a+1, N-kster+1):
                p_aux[b] = aux_mx[a, b] + p_opt_idx[b]
            mm = np.min(p_aux[a+1:N-kster+1])
            ix = np.argmin(p_aux[a+1:N-kster+1])
            p_opt_idx[a] = mm
            opt_pals[kster, a] = a + ix[0]

    # restore optimal decisions
    opt_part = np.zeros((1, K))
    opt_part[0] = opt_pals[K,0]
    for kster in range(K-1, 0, -1):
        opt_part[K-kster+1] = opt_pals[kster, opt_part[K-kster]]

    # find initial conditions
    opt_part = np.array([1, opt_part, N+1])




@pytest.mark.parametrize("x, mu, sigma, mat_file", [
    (np.linspace(-5, 5, 100), 0, 1, "../../test_data/test_find_thr/norm_pdf_1.mat"),
    (np.linspace(300, 600, 100), -40, 90, "../../test_data/test_find_thr/norm_pdf_2.mat"),
    (np.linspace(-10, 45, 600), 10, 2, "../../test_data/test_find_thr/norm_pdf_3.mat"),
])
def test_norm_pdf(x, mu, sigma, mat_file):
    y = norm_pdf(x, mu, sigma)
    mat_res = loadmat(mat_file)
    assert_allclose(y, mat_res["y"])

