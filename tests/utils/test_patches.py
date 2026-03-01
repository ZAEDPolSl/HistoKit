import os
import cv2
import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from src.histo_kit.grand_qc.artifacts import Artifact
from src.histo_kit.grand_qc.dataset import get_patch_grid
from src.histo_kit.utils.patches import merge_patches, patch_wsi
from src.histo_kit.utils.wsi import read_masked_region, get_regions_location, load_wsi_mag
from openslide import OpenSlide

ROOT = Path(__file__).parent.parent.parent

@pytest.mark.skip_ci
@pytest.mark.parametrize("desired_mag,patch_size, save_folder, bg_percent, overlap, extract_type", [
    (5, 256, "out_5_256_0.90_reflect", 0.05, 0.9, "reflect"),
])
def test_patch_image(desired_mag,patch_size, save_folder, bg_percent, overlap, extract_type):
    path = "/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425.svs"
    mask_path = np.load("/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425_mask_all.npz", allow_pickle=True)
    region_idx = 0
    wsi = OpenSlide(path)
    region = read_masked_region(wsi, mask_path, region_idx, desired_mag, notation="python", allow_list=(Artifact.NORM, Artifact.BG_MODEL), tol=1e-3)
    Image.fromarray(region).save("region_masked.png")
    patch_wsi(region, patch_size, save_folder, bg_percent, overlap, extract_type)

@pytest.mark.skip_ci
def test_read_region():
    path = "/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425.svs"
    mask_path = np.load("/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425_mask_all.npz", allow_pickle=True)
    region_idx = 0
    desired_mag = 1
    wsi = OpenSlide(path)
    Image.fromarray(read_masked_region(wsi, mask_path, region_idx, desired_mag, notation="python", allow_list=(Artifact.NORM, Artifact.BG_MODEL), tol=1e-3)).save("region_masked.png")


@pytest.mark.skip_ci
@pytest.mark.parametrize("patches_folder, scale_factor, alpha", [
(f"{ROOT}/test_data/test_postprocessing/out_5_256_0.90_reflect", 0.5, 0.2)])
def test_merge_patches(patches_folder, scale_factor, alpha):
    patch_names = os.listdir(patches_folder)
    a_s_1 = np.sort(np.random.uniform(0, 0.3, int(len(patch_names)/2)))
    a_s_2 = np.sort(np.random.uniform(0.8, 1, len(patch_names) - int(len(patch_names)/2)))
    a_s = np.concatenate([a_s_1, a_s_2])
    attention_scores = dict(zip(patch_names, a_s))
    overlay, attention_map_rgb, attention_map = merge_patches(patches_folder, attention_scores, scale_factor, alpha)
    overlay.save("../../test_data/test_postprocessing/overlay.png")
    attention_map_rgb.save("../../test_data/test_postprocessing/attention_map.png")

def test_get_patches():

    test_arr = np.zeros((100, 100))

    test_arr[10:20, 10:20] = 1
    test_arr[40:60, 70:90] = 1

    region_list = get_regions_location(test_arr)
    bbox_gt = [[10, 10, 20, 20], [40, 70, 60, 90]]

    assert sorted(region_list) == sorted(bbox_gt)

@pytest.mark.skip_ci
def test_get_grid():
    patch_size = 256
    overlap_gt = 0.9

    bg = np.array(Image.open(f"{ROOT}/test_data/test_utils/test_patches/bg_test.png").convert("1"))
    region_list = get_regions_location(bg)
    coords = get_patch_grid(region_list, patch_size=patch_size, overlap=overlap_gt)

    p_w = coords["x_end"][0] - coords["x_start"][0]
    p_h = coords["y_end"][0] - coords["y_start"][0]

    stride = coords["y_start"][1] - coords["y_start"][0]
    overlap = round(1 - stride/patch_size, 1)

    assert len(region_list) == 3
    assert p_w == patch_size
    assert p_h == patch_size
    assert overlap == overlap_gt

@pytest.mark.skip_ci
def test_vis_grid():
    patch_size = 256
    overlap_gt =0
    # load slide
    slide = OpenSlide("/mnt/data/Tmp/jmerta/article_vis/svs/TCGA-AP-A0LG-01Z-00-DX1.85966119-6AB0-43FD-88E8-D260059F9F4C.svs")

    # rescale region
    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(slide, 2.5, allow_upscaling=True)
    path = "/mnt/data/Tmp/jmerta/article_vis/masks/TCGA-AP-A0LG-01Z-00-DX1.85966119-6AB0-43FD-88E8-D260059F9F4C.mat"
    from PIL import Image
    import scipy.io as sio
    region = np.array(region)
    bg = sio.loadmat(path)["mask_bg"].astype(np.uint8)*255
    bg_art = Image.open("/mnt/data/Tmp/jmerta/article_vis/grandqc_overlay_vis/color_map_TCGA-AP-A0LG-01Z-00-DX1.85966119-6AB0-43FD-88E8-D260059F9F4C.png")
    bg_art = bg_art.resize((bg.shape[1], bg.shape[0]), Image.NEAREST)
    bg_art = np.array(bg_art)[:,:,0]
    bg_art[bg_art != 128] = 0
    bg_art[bg_art==128] = 1

    Image.fromarray(bg_art.astype(np.uint8)*255).save("bg_art.png")
    region_list, images_list = get_regions_location(bg)
    bg_rgb = np.array(Image.open(f"{ROOT}/test_data/test_utils/test_patches/bg_test.png").convert("RGB"))

    for r in region_list:
        y_min, x_min, y_max, x_max = r
        cv2.rectangle(
            bg_rgb,
            (x_min, y_min),
            (x_max, y_max),
            color=(255, 0, 255),
            thickness=2
        )
    cv2.imwrite("bbox_vis.png", bg_rgb)

    coords = get_patch_grid(region_list, patch_size=patch_size, overlap=overlap_gt)
    region = region*bg_art[...,None]
    for x_s, y_s, x_e, y_e in zip(coords["x_start"], coords["y_start"], coords["x_end"], coords["y_end"]):
        if bg_art[y_s:y_e, x_s:x_e].sum() <50:
            continue
        cv2.rectangle(
            region,
            (max(0, x_s), max(0, y_s)),
            (min(region.shape[1], x_e), min(region.shape[0], y_e)),
            color=(0, 255, 0),
            thickness=8
        )
    Image.fromarray(region).save("region.png")















