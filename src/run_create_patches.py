import glob
import os
import argparse
import numpy as np
from openslide import OpenSlide
from tqdm import tqdm
from src.histo_kit.grand_qc.artifacts import Artifact
from src.histo_kit.patches_extraction.patches_extractor import PatchesExtractor
from src.histo_kit.patches_extraction.preprocessing import convert_mask_grandqc
from src.histo_kit.utils.file_utils import get_basename
from src.histo_kit.utils.wsi import load_wsi_mag, read_region

"""
Script for dividing image into patches
"""

parser = argparse.ArgumentParser()

## Where to find slides and masks and where to save patches
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='/mnt/data/Datasets/HE_data/Labaj_UCEC/SVS/05_2024/')
parser.add_argument('--masks_dir', type=str, help='Input directory with masks created by grandQC for detected regions', default='/mnt/data/Datasets/HE_data/Labaj_UCEC/GRAND_QC/05_2024/')
parser.add_argument("--wsi_file", type=str, help="File extension for WSIs", default='.svs')
parser.add_argument('--masks_file', type=str, help='File extension for masks created by grandQC for detected regions', default='.mat')
parser.add_argument("--out_dir", type=str, help="Output directory for patches", default="/mnt/data/Tmp/jmerta/HE/test_data/test_patches/patches")
parser.add_argument('--workers', help="Number of workers used to process images in parallel.", default=10, type=int,choices=range(1, os.cpu_count() + 1))

# Patch extraction parameters
parser.add_argument('--extraction_mag', help="Magnification.", type=int, default=20)
parser.add_argument('--mode', help="Patch extraction mode", type=str, default=0)
parser.add_argument('--overlap', help="Overlap between patches - value from 0 (no overlap) to less than 1 (almost full overlap)", default=0.9, type=float)
parser.add_argument('--desired_mag', help="Magnification.", type=int, default=20)
parser.add_argument('--patch_size', help="Size of the patch", default=256, type=int)
parser.add_argument('--bg_percent', help="Percent of background pixels allowed (from 0 to 1)", default=0.9, type=float)
parser.add_argument('--allow_upscaling', help="Allow upscaling if desired resolution is not available (for instance when we want to downscale an image from 20x to 40x and only 20x is available.", default=True, type=bool)
args = parser.parse_args()

if __name__ == "__main__":

    args = parser.parse_args()

    # get slides names
    if args.wsi_dir is not None:
        all_slides = glob.glob(os.path.join(args.wsi_dir, '*.svs'))
    elif args.wsi_file is not None:
        with open(args.files_to_process, 'r') as f:
            all_slides = [line.strip() for line in f.readlines()]
    else:
        print("Please provide either wsi_dir or files_to_process argument.")
        exit(1)

    print(f"Found {len(all_slides)} slides to process.\n")

    # get masks if provided
    if args.masks_dir is not None:
        all_masks = glob.glob(os.path.join(args.wsi_dir, '*.svs'))
    elif args.masks_file is not None:
        with open(args.files_to_process, 'r') as f:
            all_masks = [line.strip() for line in f.readlines()]
    else:
        print("No masks provided, patches will be extracted from the whole slide.")

    print(f"Found {len(all_masks)}.\n")

    for slide_path in tqdm(all_slides):
        try:
            slide = OpenSlide(slide_path)
            slide_basename = get_basename(slide_path)

            # find corresponding mask
            mask = [m for m in all_masks if get_basename(m) == slide_basename]
            mask_path = mask[0] if len(mask) > 0 else None

            # create output folder for patches
            out_folder_slide = os.path.join(args.out_dir, slide_basename)
            os.makedirs(out_folder_slide, exist_ok=True)


            wsi_path = "/mnt/data/Datasets/Compass/HE/4-13_he_2.svs"

            mask_bg = "/mnt/data/Tmp/jmerta/HE-masks-compass_30_11_2025/masks/4-13_he_2.mat"
            mask_bg = sio.loadmat(mask_bg)

            mask_qc = "/mnt/data/Tmp/jmerta/HE-masks-compass_30_11_2025/masks_grandqc/4-13_he_2.mat"
            mask_qc = sio.loadmat(mask_qc)

            wsi = openslide.OpenSlide(wsi_path)

            region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, args.desired_mag, allow_upscaling=True)
            reg_list, bbox = convert_mask_grandqc(mask_qc, np.array(region), args.desired_mag,
                                                  art_include=[Artifact.NORM.value, Artifact.BG_MODEL.value],
                                                  mode="wsi")

            transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])

            extractor = PatchesExtractor(reg_list, "wsi1", out_dir="test_extract_wsi", out_dir_aug="aug",
                                         patch_size=256, overlap=0.5, aug=transform)
            extractor.extract_patches()







        except Exception as e:
            print(f"Error processing slide {slide_path}: {e}")









