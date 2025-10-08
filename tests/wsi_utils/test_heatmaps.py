import os
import numpy as np
import pytest
from PIL import Image
from openslide import OpenSlide, OpenSlideUnsupportedFormatError
from openslide.lowlevel import get_best_level_for_downsample

from src.wsi_utils.heatmaps import load_wsi_mag, patch_wsi
from math import ceil, floor
from tqdm import tqdm
import cv2


@pytest.mark.parametrize("wsi_path, patch_size, overlap, mirroring_type, mag, out_folder", [
("/mnt/data/Tmp/jmerta/HE/test_data/region.png",256, 90, "skip", 10, "out"),
])
def test_patch_image(wsi_path, patch_size, overlap, mirroring_type, mag, out_folder):
    region = np.array(Image.open(wsi_path).convert("RGB"))
    patch_wsi(region, patch_size, "out", 0.05, overlap=0, extract_type="valid")

def test_read_region():
    region_idx = 0
    desired_mag = 20

    wsi = OpenSlide("/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425.svs")
    npz_file = np.load("/mnt/data/Tmp/jmerta/HE/test_data/test_utils/SS45212_R0A10F2A_190425_mask_all.npz",
                       allow_pickle=True)

    bbox = npz_file["tiss_stats"][region_idx]
    scale_val = npz_file["scale_val"] / desired_mag
    mag_l0 = float(wsi.properties["openslide.objective-power"])
    desired_ratio = mag_l0 / desired_mag
    bbox = (bbox * scale_val).astype(int)

    level = wsi.get_best_level_for_downsample(desired_ratio)
    region = wsi.read_region((bbox[0], bbox[1]), level, (bbox[2], bbox[3])).convert("RGB")

    w0, h0 = wsi.level_dimensions[0]
    des_w, des_h = int(w0 / desired_ratio), int(h0 / desired_ratio)

    if desired_ratio not in wsi.level_downsamples:
        region = region.resize((des_h, des_w), Image.Resampling.LANCZOS)

    mask_art = npz_file["mask_art"][region_idx]
    mask_art.resize((des_h, des_w), Image.Resampling.NEAREST)

    # tissue - white, bg and artifacts - black

    region = np.array(region) * np.repeat(mask_art[:, :, np.newaxis], 3, axis=2)
    region[region == (0, 0, 0)] = (255, 255, 255)
    Image.fromarray(region).imshow()









@pytest.mark.parametrize("wsi, desired_mag, rescale_method, verbose, allow_upscaling, res", [
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 10, Image.BICUBIC, True, True, "Desired resolution is not available, image will be rescaled from the highest magnification available."),
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 5, Image.LANCZOS, True, True, "Desired magnification is available"),
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 20, Image.BICUBIC, True, True, "Desired magnification is available"),
(OpenSlide("../../test_data/tissue_seg/wsi/C3N-00339-23.svs"), 40, Image.BICUBIC, True, True, "Desired magnification is available"),
])
def test_rescale_wsi(wsi, desired_mag, rescale_method, verbose, allow_upscaling, res):
    region, scale_val, info, mpp, ratio  = load_wsi_mag(wsi, desired_mag, rescale_method, verbose, allow_upscaling)
    assert info == res











