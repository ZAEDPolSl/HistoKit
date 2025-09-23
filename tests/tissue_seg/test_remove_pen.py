import numpy as np
import pytest
from scipy.io import loadmat, savemat
from PIL import Image
from numpy.testing import assert_array_equal, assert_allclose
from src.tissue_seg.find_thr import get_pixel_distribution, norm_pdf, gmm_init_dp_hist, EM_iter_hist, find_thr, \
    GaMRed_hist


@pytest.mark.parametrize("image_path, mat_file", [
    ("../../test_data/test_find_thr/region_1.tif", "../../test_data/test_find_thr/pixel_distribution_1.mat"),
    ("../../test_data/test_find_thr/region_2.tif", "../../test_data/test_find_thr/pixel_distribution_2.mat"),
    ("../../test_data/test_find_thr/region_3.tif", "../../test_data/test_find_thr/pixel_distribution_3.mat")
])
def test_remove_black_pen(image_path, mat_file):
    img = Image.open(image_path)
    img_np = np.array(img)
    R, G, B = get_pixel_distribution(img_np)
    mat_res = loadmat(mat_file)

    assert_array_equal(R.reshape(1, -1), mat_res["R"])
    assert_array_equal(G.reshape(1, -1), mat_res["G"])
    assert_array_equal(B.reshape(1, -1), mat_res["B"])

@pytest.mark.parametrize("image_path, mat_file", [
    ("../../test_data/test_find_thr/region_1.tif", "../../test_data/test_find_thr/pixel_distribution_1.mat"),
    ("../../test_data/test_find_thr/region_2.tif", "../../test_data/test_find_thr/pixel_distribution_2.mat"),
    ("../../test_data/test_find_thr/region_3.tif", "../../test_data/test_find_thr/pixel_distribution_3.mat")
])
def test_remove_green_pen(image_path, mat_file):
    img = Image.open(image_path)
    img_np = np.array(img)
    R, G, B = get_pixel_distribution(img_np)
    mat_res = loadmat(mat_file)

    assert_array_equal(R.reshape(1, -1), mat_res["R"])
    assert_array_equal(G.reshape(1, -1), mat_res["G"])
    assert_array_equal(B.reshape(1, -1), mat_res["B"])