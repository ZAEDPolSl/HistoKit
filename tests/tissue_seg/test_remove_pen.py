import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.io import loadmat
from PIL import Image
from numpy.testing import assert_array_equal
from src.tissue_seg.remove_pen import remove_pen, get_strel_disk


@pytest.mark.parametrize("img_path, thr, mat_file", [
("../../test_data/test_find_thr/region_1.tif", {'B': 235.34549053454907, 'G': 233.32283033228305, 'R': 235.17701201770123}, "../../test_data/test_find_thr/remove_black_pen_1.mat"),
("../../test_data/test_find_thr/region_2.tif", {'B': 227.69594876959488, 'G': 225.96026559602657, 'R': 228.62759136275915}, "../../test_data/test_find_thr/remove_black_pen_2.mat"),
("../../test_data/test_find_thr/region_3.tif", {'B': 227.10684771068478, 'G': 225.6179280617928, 'R': 228.38039433803945}, "../../test_data/test_find_thr/remove_black_pen_3.mat"),
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


def test_get_strel_disk():

    strel = get_strel_disk(5)

    strel_gt = np.array([
    [False, False, True,  True,  True,  True,  True,  False, False],
    [False, True,  True,  True,  True,  True,  True,  True,  False],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [True,  True,  True,  True,  True,  True,  True,  True,  True ],
    [False, True,  True,  True,  True,  True,  True,  True,  False],
    [False, False, True,  True,  True,  True,  True,  False, False],
])
    assert_array_equal(strel, strel_gt)