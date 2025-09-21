import numpy as np
import pytest
from scipy.io import loadmat, savemat
from PIL import Image
from numpy.testing import assert_array_equal, assert_allclose
from src.tissue_seg.find_thr import get_pixel_distribution, norm_pdf, gmm_init_dp_hist


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

    x = np.arange(256)
    y = channels[channel].squeeze()
    alpha, mu, sigma = gmm_init_dp_hist(x, y, K)
    mat_res = loadmat(mat_file)
    assert_allclose(alpha, mat_res["alpha"])
    assert_allclose(mu, mat_res["mu"])
    assert_allclose(sigma, mat_res["sigma"])


@pytest.mark.parametrize("x, mu, sigma, mat_file", [
    (np.linspace(-5, 5, 100), 0, 1, "../../test_data/test_find_thr/norm_pdf_1.mat"),
    (np.linspace(300, 600, 100), -40, 90, "../../test_data/test_find_thr/norm_pdf_2.mat"),
    (np.linspace(-10, 45, 600), 10, 2, "../../test_data/test_find_thr/norm_pdf_3.mat"),
])
def test_norm_pdf(x, mu, sigma, mat_file):
    y = norm_pdf(x, mu, sigma)
    mat_res = loadmat(mat_file)
    assert_allclose(y, mat_res["y"])

