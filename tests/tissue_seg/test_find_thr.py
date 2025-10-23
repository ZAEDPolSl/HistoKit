import os
import numpy as np
import pytest
from scipy.io import loadmat
from PIL import Image
from numpy.testing import assert_array_equal, assert_allclose
from src.histo_kit.tissue_seg.find_thr import get_pixel_distribution, GaMRed_hist, EM_iter_hist, gmm_init_dp_hist, \
    norm_pdf, find_thr, get_thr_image, two_step_otsu
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

@pytest.mark.skip_ci
@pytest.mark.parametrize("image_path, mat_file", [
    (f"{ROOT}/test_data/tissue_seg/test_find_thr/region_1.tif", f"{ROOT}/test_data/tissue_seg/test_find_thr/pixel_distribution_1.mat"),
    (f"{ROOT}/test_data/tissue_seg/test_find_thr/region_3.tif", f"{ROOT}/test_data/tissue_seg/test_find_thr/pixel_distribution_3.mat")
])
def test_find_pixel_distribution(image_path, mat_file):
    img = Image.open(image_path)
    img_np = np.array(img)
    R, G, B = get_pixel_distribution(img_np)
    mat_res = loadmat(mat_file)

    assert_array_equal(R.reshape(1, -1), mat_res["R"])
    assert_array_equal(G.reshape(1, -1), mat_res["G"])
    assert_array_equal(B.reshape(1, -1), mat_res["B"])

@pytest.mark.skip_ci
@pytest.mark.parametrize("channel, K, draw, SW, mat_file", [
    ("R", 2, False, 5, f"{ROOT}/test_data/tissue_seg/test_find_thr/GaMRed_hist_1.mat"),
    ("B", 2, False, 5, f"{ROOT}/test_data/tissue_seg/test_find_thr/GaMRed_hist_3.mat")
])
def test_GaMRed_hist(channel, K, draw, SW, mat_file):
    img = Image.open(f"{ROOT}/test_data/tissue_seg/test_find_thr/region_3.tif")
    img_np = np.array(img)
    distribution = get_pixel_distribution(img_np)
    channels = {"R": distribution[0],
                "G": distribution[1],
                "B": distribution[2]}
    x = np.arange(256)
    y = channels[channel]
    thr, bic, stats = GaMRed_hist(x, y, K, draw, SW)
    mat_res = loadmat(mat_file)

    assert_allclose(thr, mat_res["thr"], rtol=1e-7)
    assert_allclose(bic, mat_res["bic"], rtol=1e-7)
    assert_allclose(stats['alpha'].reshape(1, -1), mat_res["alpha"], rtol=1e-7)
    assert_allclose(stats['mu'].reshape(1, -1), mat_res["mu"], rtol=1e-7)
    assert_allclose(stats['K'], mat_res["K"], rtol=1e-7)
    assert_allclose(stats['sigma'].reshape(1, -1), mat_res["sigma"], rtol=1e-7)
    assert_allclose(stats['logL'], mat_res["logL"], rtol=1e-7)


@pytest.mark.skip_ci
@pytest.mark.parametrize("channel, alpha, mu, sig, SW, mat_file", [
    ("R",np.array([0.3060, 0.6940]),np.array([187.8922, 242.0600]),np.array([33.1638, 0.7579]),5, f"{ROOT}/test_data/tissue_seg/test_find_thr/EM_iter_hist_1.mat"),
    ("G",np.array([0.300,0.263,0.435]),np.array([1.26e+02,2.40e+02,2.43e+02]),np.array([42.12,2.98,0.23]),5, f"{ROOT}/test_data/tissue_seg/test_find_thr/EM_iter_hist_2.mat"),
    ("B",np.array([0.0003,    0.2998,    0.6138,    0.0859,    0.0002]), np.array([61.1354,  172.2483,  241.6588,  243.1836,  252.4216]),
     np.array([7.9579,   30.8136,    1.0814,    0.8200,    0.4007]),4, f"{ROOT}/test_data/tissue_seg/test_find_thr/EM_iter_hist_3.mat"),
])
def test_EM_iter_hist(channel, alpha, mu, sig, SW, mat_file):
    img = Image.open(f"{ROOT}/test_data/tissue_seg/test_find_thr/region_3.tif")
    img_np = np.array(img)
    distribution = get_pixel_distribution(img_np)
    channels = {"R": distribution[0],
                "G": distribution[1],
                "B": distribution[2]}
    x = np.arange(256)
    y = channels[channel]
    pp_est, mu_est, sig_est, logL = EM_iter_hist(x, y, alpha, mu, sig, SW)
    mat_res = loadmat(mat_file)
    assert_allclose(np.sort(pp_est.reshape(1, -1)), np.sort(mat_res["pp_est"]))
    assert_allclose(mu_est.reshape(1, -1), mat_res["mu_est"])
    assert_allclose(sig_est.reshape(1, -1), mat_res["sig_est"])
    assert_allclose(logL.reshape(1, -1), mat_res["logL"])

@pytest.mark.skip_ci
@pytest.mark.parametrize("channel, K, mat_file", [
    ("R",2, f"{ROOT}/test_data/tissue_seg/test_find_thr/gmm_init_dp_hist_1.mat"),
    ("G",3, f"{ROOT}/test_data/tissue_seg/test_find_thr/gmm_init_dp_hist_2.mat"),
    ("B",10, f"{ROOT}/test_data/tissue_seg/test_find_thr/gmm_init_dp_hist_3.mat")
])
def test_gmm_init_dp_hist(channel, K, mat_file):
    img = Image.open(f"{ROOT}/test_data/tissue_seg/test_find_thr/region_3.tif")
    img_np = np.array(img)
    distribution = get_pixel_distribution(img_np)
    channels = {"R": distribution[0],
                "G": distribution[1],
                "B": distribution[2]}

    x = np.arange(256)
    y = channels[channel]
    alpha, mu, sigma = gmm_init_dp_hist(x, y, K)
    mat_res = loadmat(mat_file)
    assert_allclose(alpha.reshape(1, -1), mat_res["alpha"])
    assert_allclose(mu.reshape(1, -1), mat_res["mu"])
    assert_allclose(sigma.reshape(1, -1), mat_res["sigma"])


@pytest.mark.parametrize("x, mu, sigma, mat_file", [
    (np.linspace(-5, 5, 100), 0, 1, f"{ROOT}/test_data/tissue_seg/test_find_thr/norm_pdf_1.mat"),
    (np.linspace(300, 600, 100), -40, 90, f"{ROOT}/test_data/tissue_seg/test_find_thr/norm_pdf_2.mat"),
    (np.linspace(-10, 45, 600), 10, 2, f"{ROOT}/test_data/tissue_seg/test_find_thr/norm_pdf_3.mat"),
])
def test_norm_pdf(x, mu, sigma, mat_file):
    y = norm_pdf(x, mu, sigma)
    mat_res = loadmat(mat_file)
    assert_allclose(y, mat_res["y"])


@pytest.mark.parametrize("data, alpha, mi, sigma, idx, draw, thr_gt", [
    (np.arange(1, 254), np.array([0.2937, 0.7063]), np.array([186.3796,  241.7462]), np.array([33.0828,    5.0000]), np.array([0, 1]),False,  228.38027793),
    (np.arange(1, 254), np.array([0.3048, 0.6952]), np.array([128.3077,  242.1855]), np.array([43.5041,    5.0000]), np.array([0, 1]),False,  225.6180300618),
    (np.arange(1, 254), np.array([0.2974, 0.7026]), np.array([171.8093,  241.7266]), np.array([31.0049,    5.0000]), np.array([0, 1]),False,  227.10664461066)
])
def test_find_thr(data, alpha, mi, sigma, idx, draw, thr_gt):
    thr = find_thr(data, alpha, mi, sigma, idx, draw)
    assert_allclose(thr, thr_gt)

@pytest.mark.skip_ci
@pytest.mark.parametrize("img_path, mat_file", [
    (f"{ROOT}/test_data/tissue_seg/regions/region_1.tif", f"{ROOT}/test_data/tissue_seg/test_find_thr/get_thr_image_1.mat"),
])
def test_get_thr_image(img_path, mat_file):
    img = Image.open(img_path)
    img_np = np.array(img)
    thr, R, G, B = get_thr_image(img_np)

    mat_res = loadmat(mat_file)
    assert_allclose(thr["R"], mat_res["R"])
    assert_allclose(thr["G"], mat_res["G"])
    assert_allclose(thr["B"], mat_res["B"])

@pytest.mark.skip_ci
@pytest.mark.parametrize("img_path, mat_file", [
    (f"{ROOT}/test_data/tissue_seg/regions/region_1.tif", f"{ROOT}/test_data/tissue_seg/test_find_thr/two_step_otsu_1.mat"),
])
def test_two_step_otsu(img_path, mat_file):
    img = Image.open(img_path)
    img_np = np.array(img)
    R, G, B = get_pixel_distribution(img_np)

    thr= {"R": two_step_otsu(R), "G": two_step_otsu(G),
          "B": two_step_otsu(B)}
    mat_res = loadmat(mat_file)

    assert_allclose(thr["R"], mat_res["R"][0, 0])
    assert_allclose(thr["G"], mat_res["G"][0, 0])
    assert_allclose(thr["B"], mat_res["B"][0, 0])



