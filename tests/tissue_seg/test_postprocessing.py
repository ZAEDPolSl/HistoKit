import numpy as np
import pytest
from scipy.io import loadmat
from PIL import Image
from numpy.testing import assert_array_equal
from src.histo_kit.tissue_seg.postprocessing import remove_black_pen, remove_gray_stains, remove_small_objects
from src.histo_kit.utils.apply_mask import apply_mask
from src.histo_kit.utils.matlab2python import get_wsi_ind_matlab, get_strel_disk
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


@pytest.mark.parametrize("radius, strel_gt", [
(2, np.array([
    [False, False, True, False, False],
    [False, True, True, True, False],
    [True, True, True, True, True],
    [False, True, True, True, False],
    [False, False, True, False, False]
])),
(5, np.array([
    [False, False, True,  True,  True,  True,  True,  False, False],
    [False, True,  True,  True,  True,  True,  True,  True,  False],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [False, True,  True,  True,  True,  True,  True,  True,  False],
    [False, False, True,  True,  True,  True,  True,  False, False],
])),
    (3, np.array([
    [True, True, True, True, True],
    [True, True, True, True, True],
    [True, True, True, True, True],
    [True, True, True, True, True],
    [True, True, True, True, True]
])),
    (6, np.array([[False, False, True,  True,  True,  True,  True,  True,  True,  False, False],
    [False, True,  True,  True,  True,  True,  True,  True,  True,  True,  False],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [False, True,  True,  True,  True,  True,  True,  True,  True,  True,  False],
    [False, False, True,  True,  True,  True,  True,  True,  True,  False, False]])),
    (9, np.array([
    [False, False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False, False, False],
    [False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False, False],
    [False, False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False],
    [False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False],
    [False, False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True, False, False],
    [False, False, False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,   False, False, False],
    [False, False, False, False, True,  True,  True,  True,  True,  True,  True,  True, True,  False, False, False, False]
]))

])
def test_get_strel_disk(radius, strel_gt):
    strel = get_strel_disk(radius)
    assert_array_equal(strel, strel_gt)

@pytest.mark.skip_ci
@pytest.mark.parametrize("img_path,inv, mat_file", [
(f"{ROOT}/test_data/tissue_seg/regions/region_3.tif",True, f"{ROOT}/test_data/tissue_seg/test_postprocessing/apply_mask_3.mat"),
(f"{ROOT}/test_data/tissue_seg/regions/region_3.tif",False, f"{ROOT}/test_data/tissue_seg/test_postprocessing/apply_mask_4.mat"),
])
def test_apply_mask(img_path, inv, mat_file):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_np = np.array(img)
    mat_res = loadmat(mat_file)
    img_res = apply_mask(img_np, mat_res['mask'], inv)
    assert_array_equal(img_res, mat_res["img_res"])

@pytest.mark.parametrize("mat_file", [
    f"{ROOT}/test_data/tissue_seg/test_postprocessing/remove_small_objects_1.mat",
])
@pytest.mark.repeat(5)
def test_remove_small_objects(mat_file):
    mat_res = loadmat(mat_file)
    mask_res = remove_small_objects(mat_res['mask'])
    Image.fromarray(mask_res).show()
    assert_array_equal(mask_res, mat_res["mask_res"].astype(bool))

@pytest.mark.skip_ci
@pytest.mark.parametrize("img_path, mat_file", [
(f"{ROOT}/test_data/tissue_seg/regions/region_1.tif", f"{ROOT}/test_data/tissue_seg/test_postprocessing/remove_grey_stains_1.mat"),
(f"{ROOT}/test_data/tissue_seg/regions/region_1.tif", f"{ROOT}/test_data/tissue_seg/test_postprocessing/remove_grey_stains_4.mat"),
])
def test_remove_grey_stains(img_path, mat_file):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_np = np.array(img)
    mat_res = loadmat(mat_file)
    mask_init = mat_res['mask_init']
    if mask_init.shape == (0,0):
        mask_init=None
    mask_res = remove_gray_stains(img_np, mask_init)
    diff_fraction = np.mean(mask_res != mat_res["mask"])
    diff_num = np.sum(mask_res != mat_res["mask"])
    print(f" Fraction of mismatched elements: {diff_fraction:.8f} - Number of pixels mismatched: {diff_num}")
    assert diff_fraction < 10e-7

@pytest.mark.skip_ci
@pytest.mark.parametrize("svs_path, ind_gt", [
(f"{ROOT}/test_data/tissue_seg/wsi/wsi_1.svs", [1, 3, 4, 5]),
])
def test_get_wsi_ind_matlab(svs_path, ind_gt):
    ind = get_wsi_ind_matlab(svs_path)
    assert_array_equal(ind, ind_gt)


