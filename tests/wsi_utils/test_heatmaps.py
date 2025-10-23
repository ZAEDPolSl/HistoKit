import os
import numpy as np
import pytest
from PIL import Image
from openslide import OpenSlide
from src.histo_kit.grand_qc.artifacts import Artifact
from src.histo_kit.wsi_utils.patches import read_region, patch_wsi, load_wsi_mag, merge_patches

skip_openslide = os.getenv("CI", "").lower() == "true"
if not skip_openslide:
    from openslide import OpenSlide

@pytest.mark.skipif(os.getenv("CI")=="true", reason="Large tissue files not uploaded to CI")
def bimodal_normal_vector(n, mean1=0.3, mean2=0.7, std1=0.08, std2=0.08):
    n1 = n // 2
    n2 = n - n1
    data1 = np.random.normal(mean1, std1, n1)
    data2 = np.random.normal(mean2, std2, n2)
    data = np.concatenate([data1, data2])
    data = np.clip(data, 0, 1)
    data = np.sort(data)
    return data

@pytest.mark.skipif(os.getenv("CI")=="true", reason="Large tissue files not uploaded to CI")
@pytest.mark.parametrize("desired_mag,patch_size, save_folder, bg_percent, overlap, extract_type", [
    (5, 256, "out_5_256_0.90_reflect", 0.05, 0.9, "reflect"),
])
def test_patch_image(desired_mag,patch_size, save_folder, bg_percent, overlap, extract_type):
    path = "/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425.svs"
    mask_path = np.load("/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425_mask_all.npz", allow_pickle=True)
    region_idx = 0
    wsi = OpenSlide(path)
    region = read_region(wsi, mask_path, region_idx, desired_mag, notation="python", allow_list=(Artifact.NORM, Artifact.BG_MODEL), tol=1e-3)
    Image.fromarray(region).save("region_masked.png")
    patch_wsi(region, patch_size, save_folder, bg_percent, overlap, extract_type)

@pytest.mark.skipif(os.getenv("CI")=="true", reason="Large tissue files not uploaded to CI")
def test_read_region():
    path = "/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425.svs"
    mask_path = np.load("/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425_mask_all.npz", allow_pickle=True)
    region_idx = 0
    desired_mag = 1
    wsi = OpenSlide(path)
    Image.fromarray(read_region(wsi, mask_path, region_idx, desired_mag, notation="python", allow_list=(Artifact.NORM, Artifact.BG_MODEL), tol=1e-3)).save("region_masked.png")


@pytest.mark.skipif(os.getenv("CI")=="true", reason="Large tissue files not uploaded to CI")
@pytest.mark.parametrize("wsi, desired_mag, rescale_method, verbose, allow_upscaling, res", [
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 10, Image.BICUBIC, True, True, "Desired resolution is not available, image will be rescaled from the highest magnification available."),
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 5, Image.LANCZOS, True, True, "Desired magnification is available"),
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 20, Image.BICUBIC, True, True, "Desired magnification is available"),
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 40, Image.BICUBIC, True, True, "Desired magnification is available"),
])
def test_rescale_wsi(wsi, desired_mag, rescale_method, verbose, allow_upscaling, res):
    region, scale_val, info, mpp, ratio  = load_wsi_mag(wsi, desired_mag, rescale_method, verbose, allow_upscaling)
    assert info == res

@pytest.mark.skipif(os.getenv("CI")=="true", reason="Large tissue files not uploaded to CI")
@pytest.mark.parametrize("patches_folder, scale_factor, alpha", [
("../../test_data/test_postprocessing/out_5_256_0.90_reflect", 0.5, 0.2)])
def test_merge_patches(patches_folder, scale_factor, alpha):
    patch_names = os.listdir(patches_folder)
    a_s_1 = np.sort(np.random.uniform(0, 0.3, int(len(patch_names)/2)))
    a_s_2 = np.sort(np.random.uniform(0.8, 1, len(patch_names) - int(len(patch_names)/2)))
    a_s = np.concatenate([a_s_1, a_s_2])
    attention_scores = dict(zip(patch_names, a_s))
    overlay, attention_map_rgb, attention_map = merge_patches(patches_folder, attention_scores, scale_factor, alpha)
    overlay.save("../../test_data/test_postprocessing/overlay.png")
    attention_map_rgb.save("../../test_data/test_postprocessing/attention_map.png")











