import os

import numpy as np
import pytest
from scipy.io import loadmat
from PIL import Image
from src.histo_kit.tissue_seg.bg_segmentation import wsi_tissue_seg

@pytest.mark.skipif(os.getenv("CI")=="true", reason="Large tissue files not uploaded to CI")
@pytest.mark.parametrize("img_path, mat_file", [
    ("../../test_data/tissue_seg/regions/region_1.tif","../../test_data/tissue_seg/test_bg_segmentation/tissue_seg_1.mat"),
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
    print(f"\n Fraction of mismatched elements: {diff_fraction:.8f} - Number of pixels mismatched: {diff_num}\n")
    assert diff_fraction < 10e-2