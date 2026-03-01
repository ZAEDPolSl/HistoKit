import os
import scipy.io as sio
import numpy as np
import openslide
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.histo_kit.augmentation.base import Compose, OneOf
from src.histo_kit.augmentation.blurring import GaussianBlur
from src.histo_kit.augmentation.color_augmentation import SaltAndPepper, GaussianNoise, ColorJitter
from src.histo_kit.augmentation.rotations import RandomRotation, RandomFlip
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

    des_mag = 10
    wsi_path = "/mnt/data/Tmp/jmerta/TCGA-test-time/TCGA-AJ-A3OJ-01Z-00-DX1.5E9CF5C5-DF42-4AAC-A849-35B90E8EBCAC.svs"

    mask_bg = "/mnt/data/Tmp/jmerta/HE-masks-compass_30_11_2025/masks/4-13_he_2.mat"
    mask_bg = sio.loadmat(mask_bg)

    mask_qc = "/mnt/data/Tmp/jmerta/HE-masks-compass_30_11_2025/masks_grandqc/4-13_he_2.mat"
    mask_qc = sio.loadmat(mask_qc)

    wsi = openslide.OpenSlide(wsi_path)

    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, des_mag, allow_upscaling=True)

    reg_list, bbox = convert_mask_grandqc(mask_qc, np.array(region), des_mag,
                                          art_include=[Artifact.NORM.value, Artifact.BG_MODEL.value], mode="wsi")

    transform = Compose([
        RandomFlip(prob=0.5),
        RandomFlip(prob=0.5),
        OneOf([
            GaussianBlur(radius_range=(1, 3), prob=1.0),
            RandomRotation(prob=1.0)
        ], prob=0.7)
    ])

    extractor = PatchesExtractor(reg_list,"wsi1", out_dir="test_extract_wsi",out_dir_aug="aug22", patch_size=256, overlap=0.5, aug = transform)
    extractor.extract_patches()


def test_vis_grid():
    patch_size = 512
    overlap_gt = 0.1

    region = np.array(Image.open(f"{ROOT}/test_data/test_utils/test_patches/region_test.png"))
    bg = np.array(Image.open(f"{ROOT}/test_data/test_utils/test_patches/bg_test.png").convert("1"))
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

    for x_s, y_s, x_e, y_e in zip(coords["x_start"], coords["y_start"], coords["x_end"], coords["y_end"]):
        cv2.rectangle(
            bg_rgb,
            (max(0, x_s), max(0, y_s)),
            (min(region.shape[1], x_e), min(region.shape[0], y_e)),
            color=(0, 255, 0),
            thickness=2
        )
    cv2.imwrite(f"reg_vis.png", bg_rgb)
