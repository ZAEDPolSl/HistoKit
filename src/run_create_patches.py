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
import scipy.io as sio

"""
Script for dividing image into patches
"""

parser = argparse.ArgumentParser()

## Where to find slides and masks and where to save patches
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='/mnt/warehouse/jmerta/jwandas_data/data/svs/')
parser.add_argument('--masks_dir', type=str, help='Input directory with masks created by grandQC for detected regions', default='/mnt/warehouse/jmerta/jwandas_data/data/masks_grandqc/')
parser.add_argument("--ext", type=str, help="File extension for WSIs", default='.tif')
parser.add_argument('--masks_file', type=str, help='File extension for masks created by grandQC for detected regions', default='.mat')
parser.add_argument("--out_dir", type=str, help="Output directory for patches", default="/mnt/warehouse/jmerta/jwandas_data/data/patches_10x_256_256/")
parser.add_argument('--workers', help="Number of workers used to process images in parallel.", default=10, type=int,choices=range(1, os.cpu_count() + 1))

# Patch extraction parameters
parser.add_argument('--extraction_mag', help="Magnification.", type=int, default=10)
parser.add_argument('--overlap', help="Overlap between patches - value from 0 (no overlap) to less than 1 (almost full overlap)", default=0, type=float)
parser.add_argument('--patch_size', help="Size of the patch", default=256, type=int)
parser.add_argument('--bg_percent', help="Percent of background pixels allowed (from 0 to 1)", default=0.9, type=float)
args = parser.parse_args()

if __name__ == "__main__":

    args = parser.parse_args()

    # get slides names
    all_slides = glob.glob(os.path.join(args.wsi_dir, f'*{args.ext}'))

    print(f"Found {len(all_slides)} slides to process.\n")

    for slide_path in tqdm(all_slides):
        try:
            slide = OpenSlide(slide_path)
            slide_basename = get_basename(slide_path)
            mask_path = os.path.join(args.masks_dir, slide_basename + ".mat")
            mask = sio.loadmat(mask_path)

            region, scale_val, info, mpp_slide, ratio = load_wsi_mag(slide, args.extraction_mag, allow_upscaling=True)
            reg_list, bbox = convert_mask_grandqc(mask, np.array(region), args.extraction_mag, art_include=[Artifact.NORM.value, Artifact.BG_MODEL.value], mode="wsi")

            extractor = PatchesExtractor(reg_list, slide_basename, out_dir=args.out_dir, patch_size=256, overlap=0.5)
            extractor.extract_patches()


        except Exception as e:
            print(f"Error processing slide {slide_path}: {e}")









