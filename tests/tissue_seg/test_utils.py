import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.io import loadmat
from PIL import Image
from numpy.testing import assert_array_equal
from src.tissue_seg.utils import remove_pen, get_strel_disk, apply_mask, remove_gray_stains, remove_small_objects


@pytest.mark.parametrize("img_path, thr, mat_file", [
("../../test_data/test_utils/region_1.tif", {'B': 235.34549053454907, 'G': 233.32283033228305, 'R': 235.17701201770123}, "../../test_data/test_utils/remove_black_pen_1.mat"),
("../../test_data/test_utils/region_2.tif", {'B': 227.69594876959488, 'G': 225.96026559602657, 'R': 228.62759136275915}, "../../test_data/test_utils/remove_black_pen_2.mat"),
("../../test_data/test_utils/region_3.tif", {'B': 227.10684771068478, 'G': 225.6179280617928, 'R': 228.38039433803945}, "../../test_data/test_utils/remove_black_pen_3.mat"),
])
def test_remove_black_pen(img_path, thr, mat_file):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_np = np.array(img)
    mask = remove_pen(img_np, "black", 0.7, 0, thr, 5)
    mat_res = loadmat(mat_file)
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap="gray", interpolation="nearest")
    plt.title("Mask Python")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mat_res["mask"], cmap="gray", interpolation="nearest")
    plt.title("Mask Matlab")
    plt.axis("off")

    plt.show()
    diff_fraction = np.mean(mask.astype(int)!=mat_res["mask"])
    diff_num = np.sum(mask.astype(int)!=mat_res["mask"])
    print(f" Fraction of mismatched elements: {diff_fraction:.8f} - Number of pixels mismatched: {diff_num}")
    assert diff_fraction < 10e-5


@pytest.mark.parametrize("radius, strel_gt", [
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
    (4, np.array([
    [False, False, True,  True,  True,  False, False],
    [False, True,  True,  True,  True,  True,  False],
    [True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True ],
    [False, True,  True,  True,  True,  True,  False],
    [False, False, True,  True,  True,  False, False]
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
    (7, np.array([
    [False, False, True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False],
    [False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    [False, True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  False],
    [False, False, True,  True,  True,  True,  True,  True,  True,  True,  True,  False, False]
])),
(8, np.array([
    [False, False, False, False, True, True, True, True, True, True, True, False, False, False, False],
    [False, False, False, True, True, True, True, True, True, True, True, True, False, False, False],
    [False, False, True, True, True, True, True, True, True, True, True, True, True, False, False],
    [False, True, True, True, True, True, True, True, True, True, True, True, True, True, False],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
    [False, True, True, True, True, True, True, True, True, True, True, True, True, False, False],
    [False, False, True, True, True, True, True, True, True, True, True, True, False, False, False],
    [False, False, False, True, True, True, True, True, True, True, True, False, False, False, False],
    [False, False, False, False, True, True, True, True, True, True, False, False, False, False, False]
])),
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

@pytest.mark.parametrize("img_path,inv, mat_file", [
("../../test_data/test_utils/region_1.tif",True, "../../test_data/test_utils/apply_mask_1.mat"),
("../../test_data/test_utils/region_2.tif",True, "../../test_data/test_utils/apply_mask_2.mat"),
("../../test_data/test_utils/region_3.tif",True, "../../test_data/test_utils/apply_mask_3.mat"),
("../../test_data/test_utils/region_3.tif",False, "../../test_data/test_utils/apply_mask_4.mat"),
])
def test_apply_mask(img_path, inv, mat_file):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_np = np.array(img)
    mat_res = loadmat(mat_file)
    img_res = apply_mask(img_np, mat_res['mask'], inv)
    assert_array_equal(img_res, mat_res["img_res"])

@pytest.mark.parametrize("mat_file", [
    "../../test_data/test_utils/remove_small_objects_1.mat",
    "../../test_data/test_utils/remove_small_objects_2.mat",
    "../../test_data/test_utils/remove_small_objects_3.mat",
])
@pytest.mark.repeat(5)
def test_remove_small_objects(mat_file):
    mat_res = loadmat(mat_file)
    mask_res = remove_small_objects(mat_res['mask'])
    assert_array_equal(mask_res, mat_res["mask_res"].astype(bool))

@pytest.mark.parametrize("img_path, mat_file", [
("../../test_data/test_utils/region_1.tif", "../../test_data/test_utils/remove_grey_stains_1.mat"),
("../../test_data/test_utils/region_2.tif", "../../test_data/test_utils/remove_grey_stains_2.mat"),
("../../test_data/test_utils/region_3.tif", "../../test_data/test_utils/remove_grey_stains_3.mat"),
("../../test_data/test_utils/region_1.tif", "../../test_data/test_utils/remove_grey_stains_4.mat"),
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