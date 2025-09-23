import os
import argparse
import pandas as pd
import glob
from openslide import OpenSlide
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

from src.tissue_seg.find_thr import get_pixel_distribution, GaMRed_hist

"""
Script for tissue region detection 

Original matlab implementation is available in: github.com/WSI_TissueSeg
"""

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='../test_data/wsi/')
parser.add_argument('--out_dir', type=str, help='Output directory', default='../test_data/res/')
parser.add_argument('--fill_holes', type=bool, help='Fill holes in the tissue or not', default=False)
parser.add_argument('--ask_for_overwrite', type=bool, help='Ask for overwriting the report file', default=False)
args = parser.parse_args()


MAG_BG_DET = 2.5 # magnification for tissue detection
WSI_DIR = args.wsi_dir

# Create folders for results
BG_MASK_DIR = os.path.join(args.out_dir, 'background_masks')
BG_THRESH_DIR = os.path.join(args.out_dir, 'background_thresholds')
RAW_SMALL = os.path.join(args.out_dir, 'raw_small')
PEN_CORRECTED = os.path.join(args.out_dir, 'pen_corrected')
PEN_VIS = os.path.join(args.out_dir, 'pen_vis')
GRAND_QC = os.path.join(args.out_dir, 'grand_qc')


if not os.path.exists(BG_MASK_DIR): os.makedirs(BG_MASK_DIR)
if not os.path.exists(BG_THRESH_DIR): os.makedirs(BG_THRESH_DIR)
if not os.path.exists(RAW_SMALL): os.makedirs(RAW_SMALL)
if not os.path.exists(PEN_CORRECTED): os.makedirs(PEN_CORRECTED)
if not os.path.exists(PEN_VIS): os.makedirs(PEN_VIS)
if not os.path.exists(GRAND_QC): os.makedirs(GRAND_QC)


# Check if log file exists - ask if user want to overwrite it
REPORT_FILE = os.path.join(args.out_dir, 'report_bg_removal.csv')
if os.path.exists(REPORT_FILE) and args.ask_for_overwrite:
    ans = input(f"Do you want to remove file with logs from the previous run? (y/n): ").strip().lower()
    if ans == 'y':
        os.remove(REPORT_FILE)
        print(f'Deleted report file form the previous run: {REPORT_FILE}')
    else:
        print(f'Results will be appended to the existing report file: {REPORT_FILE}')

# get slides names
slides = glob.glob(os.path.join(WSI_DIR, '*.svs'))

# Process WSIs
print(f"Found {len(slides)} WSIs in {WSI_DIR} directory.\n")
for slide in slides:
    print(f"Processing slide: {os.path.basename(slide)}\n")
    basename = os.path.basename(slide).split('.')[0]

    # load slide
    slide = OpenSlide(slide)

    # get resizing ratio for each layer
    ratio = slide.level_downsamples

    # load image with 2.5 magnification (if it is not available - rescale it from the highest magnification)
    mag = float(slide.properties["openslide.objective-power"])  # get magnification of the 0 layer
    mag_layers = [round(mag / r, 2) for r in ratio]  # magnification of each layer

    if MAG_BG_DET in mag_layers:
        mag_idx = mag_layers.index(MAG_BG_DET)
        w, h = slide.level_dimensions[mag_idx]
        region = slide.read_region((0, 0), mag_idx, (w, h))
    else:
        mag_idx = 0  # get the highest magnification and rescale
        w, h = slide.level_dimensions[mag_idx]
        region = slide.read_region((0, 0), mag_idx, (w, h))
        scale_factor = MAG_BG_DET / mag
        region = region.resize((int(w * scale_factor), int(h * scale_factor)), Image.BICUBIC)

    scale_thumbnail = 1/MAG_BG_DET
    thumbnail = region.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    thumbnail.save(os.path.join(RAW_SMALL, f'{basename}.png'))

    # get the distribution of pixel values per color channel
    img_np = np.array(region)
    R, G, B = get_pixel_distribution(img_np)

    x = np.arange(256)
    K= 2
    SW = 5
    draw = False

    thr = np.zeros(3)
    thr[0] = GaMRed_hist(x, R, K, draw, SW)[0]
    thr[1] = GaMRed_hist(x, G, K, draw, SW)[0]
    thr[2] = GaMRed_hist(x, B, K, draw, SW)[0]

    bins = np.arange(-0.5, 255.5, 1)


    bins = np.arange(0, 255, 1)
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))

    axs[0].bar(bins, R, color='red')
    axs[0].set_title('Red')
    axs[1].bar(bins, G, color='green')
    axs[1].set_title('Green')
    axs[2].bar(bins, B, color='blue')
    axs[2].set_title('Blue')

    for ax in axs:
        ax.set_xlim(0, 255)

    axs[2].set_xlabel("Pixel value")
    axs[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    print(mag_layers)

    print(slide.level_dimensions)

    mag_idx = 3

    # read region with chosen magnification

    plt.figure(figsize=(20, 20))
    plt.imshow(region)
    plt.axis('off')
    plt.show()

    log_slide = {}
    if os.path.exists(REPORT_FILE):
        pd.DataFrame(log_slide).to_csv(REPORT_FILE, mode="a", header=False, index=False)
    else:
        pd.DataFrame(log_slide).to_csv(REPORT_FILE, mode="w", header=True, index=False)



