import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.io import loadmat
from PIL import Image
from numpy.testing import assert_array_equal
from src.tissue_seg.utils import remove_pen, get_strel_disk, apply_mask, remove_gray_stains, remove_small_objects
from src.tissue_seg.tissue_seg import wsi_tissue_seg

@pytest.mark.parametrize("img_path, mat_file", [
    ("../../test_data/test_utils/region_1.tif","../../test_data/test_tissue_seg/tissue_seg_1.mat"),
    ("../../test_data/test_utils/region_2.tif","../../test_data/test_tissue_seg/tissue_seg_2.mat"),
    ("../../test_data/test_utils/region_3.tif","../../test_data/test_tissue_seg/tissue_seg_3.mat"),
])
def test_tissue_seg(img_path, mat_file):
    mat_res = loadmat(mat_file)
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_np = np.array(img)
    res = wsi_tissue_seg(img_np)
    mask_res = res['mask']
    diff_fraction = np.mean(mask_res != mat_res["mask_res"])
    diff_num = np.sum(mask_res != mat_res["mask_res"])


    mask_res = Image.fromarray(mask_res)
    reduced_size = (mask_res.width//10, mask_res.height//10)
    mask_res = np.array(mask_res.resize(reduced_size, Image.Resampling.NEAREST)).astype(bool)
    mask_res_gt = Image.fromarray(mat_res["mask_res"])
    mask_res_gt = np.array(mask_res_gt.resize(reduced_size, Image.Resampling.NEAREST)).astype(bool)
    rgb = np.zeros((*mask_res_gt.shape, 3), dtype=np.uint8)
    rgb[mask_res & mask_res_gt] = [255, 255, 255]
    rgb[mask_res & ~mask_res_gt] = [0, 0, 255]
    rgb[~mask_res & mask_res_gt] = [255, 0, 0]
    rgb = Image.fromarray(rgb)
    rgb.save(mat_file.replace(".mat", "_res_thumbnail.png"))
    print(f"\n Fraction of mismatched elements: {diff_fraction:.8f} - Number of pixels mismatched: {diff_num}\n")
    assert diff_fraction < 10e-2