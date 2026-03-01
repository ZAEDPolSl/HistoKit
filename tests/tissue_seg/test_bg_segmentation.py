import os
from openslide import OpenSlide
import numpy as np
import openslide
import pytest
from scipy.io import loadmat
from PIL import Image
from src.histo_kit.tissue_seg.bg_segmentation import wsi_tissue_seg

from pathlib import Path

from src.histo_kit.utils.file_utils import get_basename
from src.histo_kit.utils.wsi import load_wsi_mag, get_regions_location

ROOT = Path(__file__).parent.parent.parent

@pytest.mark.skip_ci
@pytest.mark.parametrize("img_path, mat_file", [
    (f"{ROOT}/test_data/tissue_seg/regions/region_1.tif",f"{ROOT}/test_data/tissue_seg/test_bg_segmentation/tissue_seg_1.mat"),
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

@pytest.mark.skip_ci
@pytest.mark.parametrize("folder_mat, folder_svs", [
    (f"/mnt/data/Tmp/jmerta/Compass_HistoKit_test/Background_matlab/",
     f"/mnt/data/Datasets/Compass/HE/")
])
def test_tissue_seg_folder(folder_mat, folder_svs):

    mat_files = [f for f in os.listdir(folder_mat) if f.endswith('.mat')]
    des_mag = 2.5

    for mat_f in mat_files:
        mask = loadmat(os.path.join(folder_mat, mat_f))
        basename = get_basename(mat_f).removesuffix("_mask_all")
        svs_path = os.path.join(folder_svs, f"{basename}.svs")
        wsi = OpenSlide(svs_path)
        region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, des_mag, allow_upscaling=True)
        region_np = np.array(region)
        res = wsi_tissue_seg(region_np)
        res["mask"] = res["mask"].astype(np.uint8)




