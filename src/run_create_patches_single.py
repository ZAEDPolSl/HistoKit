import glob
import os
import argparse
import time
import numpy as np
from openslide import OpenSlide
from PIL import Image
import scipy.io as sio

from src.histo_kit.utils.patches import load_wsi_mag, read_region, patch_wsi

"""
Script for dividing image into patches
"""

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='/mnt/data/Datasets/HE_data/Labaj_UCEC/SVS/05_2024/')
parser.add_argument("--masks_folder", type=str, help="Input directory with masks created by grandQC for detected regions with file extension. ", default="/mnt/data/Tmp/jmerta/HE/test_data/test_patches/masks")
parser.add_argument('--out_dir', type=str, help='Output directory for patches (patches from each wsi will be saved to a separated subfolder with the name defined by wsi name)', default='/mnt/data/Tmp/jmerta/HE/test_data/test_patches/masks/')
parser.add_argument('--desired_mag', help="Magnification.", type=int, default=10)
parser.add_argument('--rescale_method', help="Method used to rescale image region if the desired resolution is not available: Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.LANCZOS, Image.NEAREST", type=int, default=Image.LANCZOS, choices=[Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS])
parser.add_argument('--workers', help="Number of workers used to process images in parallel.", default=10, type=int,choices=range(1, os.cpu_count() + 1))
parser.add_argument('--overlap', help="Overlap between patches - value from 0 (no overlap) to less than 1 (almost full overlap)", default=0.9, type=float)
parser.add_argument('--extract_type', help="Patch extraction mode, can be one of: ", default=10, type=int, choices=range(1, os.cpu_count() + 1))
parser.add_argument('--patch_size', help="Size of the patch", default=256, type=int)
parser.add_argument('--notation', help="Bounding box notation needed when you want to split regions: \"python\" or \"matlab\"", default="matlab", type=str, choices=["python", "matlab"])
parser.add_argument('--bg_percent', help="Percent of background pixels allowed (from 0 to 1)", default=0.9, type=float)
parser.add_argument('--allow_upscaling', help="Allow upscaling if desired resolution is not available (for instance when we want to downscale an image from 20x to 40x and only 20x is available.", default=True, type=bool)
parser.add_argument('--mask_available', help="Files with masks are available", default=True, type=bool)
args = parser.parse_args()

def patch_slides(slide, mask, patch_size, out_dir, bg_percent, overlap, extract_type, desired_mag, rescale_method, allow_upscaling):
    error_slides = []
    error_msgs = []
    deltas = []
    processed = []

    try:
        start = time.time()
        wsi = OpenSlide(slide)

        if not args.mask_available:
            region = load_wsi_mag(wsi, desired_mag, rescale_method, verbose = False, allow_upscaling = allow_upscaling)
        else:
            if os.path.splitext(mask)[1] == ".mat":
                mask = sio.loadmat(mask)
            elif os.path.splitext(mask)[1] == ".npz":
                mask = np.load(mask, allow_pickle=True)

            reg_num = len(mask["tiss_stats"])
            for r in range(reg_num):
                region = read_region(wsi, mask, r, desired_mag, notation = args.notation, allow_list = [], resampling_method = rescale_method)

        patch_wsi(region, patch_size, out_dir, bg_percent, overlap, extract_type)
        delta = time.time() - start
        deltas.append(delta)
        processed.append(region)
    except Exception as e:
        error_slides.append(region)
        error_msgs.append(str(e))
        deltas.append(0)
    return error_slides, error_msgs, deltas, processed

if __name__ == "__main__":

    # slide names and masks - check if each slide has a mask
    slides = sorted(glob.glob(os.path.join(args.wsi_dir, '*.svs')))
    masks = sorted(glob.glob(os.path.join(args.masks_folder, '*.mat')))

    slides_base = {os.path.splitext(os.path.basename(s))[0] for s in slides}
    masks_base = {os.path.splitext(os.path.basename(s))[0].removesuffix("_mask_all") for s in masks}

    missing_masks =  slides_base - masks_base

    if len(missing_masks)==0:
        print("All SVS have masks.")
    else:
        print(f"Missing masks for {len(missing_masks)} SVS files:")
        for mask in missing_masks:
            print(mask)

    time_start = time.time()
    log_file = os.path.join(args.out_dir, "error_files.txt")

    for m_b in masks_base:

        slide = os.path.join(args.wsi_dir, m_b + ".svs")
        mask = os.path.join(args.masks_folder, m_b + "_mask_all.mat")

        patch_slides(slide, mask, args.patch_size, args.out_dir, args.bg_percent, args.overlap, args.extract_type, args.desired_mag,
                     args.rescale_method, args.allow_upscaling)
