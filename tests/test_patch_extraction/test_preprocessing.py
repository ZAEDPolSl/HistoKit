import os
import scipy.io as sio
import numpy as np
import openslide
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.histo_kit.patches_extraction.patches_extractor import PatchesExtractor, collate_remove_none
from src.histo_kit.patches_extraction.preprocessing import convert_mask_grandqc, convert_mask_bg
from src.histo_kit.utils.wsi import load_wsi_mag
from src.histo_kit.grand_qc.artifacts import Artifact
from PIL import Image, ImageDraw


def test_convert_mask_grandqc():

    des_mag = 10
    wsi_path = "/mnt/data/Datasets/Compass/HE/4-13_he_2.svs"

    mask_bg = "/mnt/data/Tmp/jmerta/HE-masks-compass_30_11_2025/masks/4-13_he_2.mat"
    mask_bg = sio.loadmat(mask_bg)

    mask_qc = "/mnt/data/Tmp/jmerta/HE-masks-compass_30_11_2025/masks_grandqc/4-13_he_2.mat"
    mask_qc = sio.loadmat(mask_qc)

    wsi = openslide.OpenSlide(wsi_path)

    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, des_mag, allow_upscaling=True)

    reg_list, bbox = convert_mask_grandqc(mask_qc, np.array(region), des_mag,
                                          art_include=[Artifact.NORM.value, Artifact.BG_MODEL.value], mode="region")

    for idx, r in enumerate(reg_list):
        Image.fromarray(r).save(os.path.join(str(idx) + "_region.png"))

    reg_list, bbox = convert_mask_grandqc(mask_qc, np.array(region), des_mag,
                                          art_include=[Artifact.NORM.value, Artifact.BG_MODEL.value], mode="wsi")


    thickness = 3
    color = (255, 0, 0)

    for idx, r in enumerate(reg_list):
        img = Image.fromarray(r)
        draw = ImageDraw.Draw(img)

        for box in bbox:
            y0, x0, y1, x1 = map(int, box)
            rect = (x0, y0, x1, y1)

            for t in range(thickness):
                draw.rectangle([rect[0] - t, rect[1] - t, rect[2] + t, rect[3] + t], outline=color)

        out_path = os.path.join(str(idx) + "_wsi_bbox.png")
        img.save(out_path)

def test_convert_mask_bg():

    des_mag = 10
    wsi_path = "/mnt/data/Datasets/Compass/HE/4-13_he_2.svs"

    mask_bg = "/mnt/data/Tmp/jmerta/HE-masks-compass_30_11_2025/masks/4-13_he_2.mat"
    mask_bg = sio.loadmat(mask_bg)

    wsi = openslide.OpenSlide(wsi_path)

    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, des_mag, allow_upscaling=True)

    reg_list, bbox = convert_mask_bg(mask_bg, np.array(region), des_mag,
                                          art_include=[Artifact.NORM.value], mode="region")

    for idx, r in enumerate(reg_list):
        Image.fromarray(r).save(os.path.join(str(idx) + "bg_region.png"))

    reg_list, bbox = convert_mask_bg(mask_bg, np.array(region), des_mag,
                                          art_include=[Artifact.NORM.value], mode="wsi")


    thickness = 3
    color = (255, 0, 0)

    for idx, r in enumerate(reg_list):
        img = Image.fromarray(r)
        draw = ImageDraw.Draw(img)

        for box in bbox:
            y0, x0, y1, x1 = map(int, box)
            rect = (x0, y0, x1, y1)

            for t in range(thickness):
                draw.rectangle([rect[0] - t, rect[1] - t, rect[2] + t, rect[3] + t], outline=color)

        out_path = os.path.join(str(idx) + "_bg_wsi_bbox.png")
        img.save(out_path)

def test_patches_extractor():

    des_mag = 2.5
    wsi_path = "/mnt/warehouse/jmerta/jwandas_data/114S.tif"
    mask_bg = "/mnt/warehouse/jmerta/jwandas_data/114S.mat"
    mask_bg = sio.loadmat(mask_bg)
    mask_qc = "/mnt/warehouse/jmerta/jwandas_data/114S_gqc.mat"
    mask_qc = sio.loadmat(mask_qc)
    wsi = openslide.OpenSlide(wsi_path)
    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, des_mag, allow_upscaling=True)

    reg_list, bbox = convert_mask_grandqc(mask_qc, np.array(region), des_mag,
                                          art_include=[Artifact.NORM.value, Artifact.BG_MODEL.value], mode="wsi")


    extractor = PatchesExtractor(reg_list,"wsi_tiff", out_dir="test_extract_wsi",out_dir_aug="aug", patch_size=256, overlap=0.5)
    extractor.extract_patches()
