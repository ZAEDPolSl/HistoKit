import os
import argparse
import cv2
import numpy as np
import pandas as pd
import glob
from openslide import OpenSlide
import matplotlib.pyplot as plt
import PIL.Image as Image
from src.tissue_seg.tissue_seg import wsi_tissue_seg, plot_rgb_hist
from src.tissue_seg.utils import apply_mask

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
PEN_VIS = os.path.join(args.out_dir, 'pen_vis')
REMOVAL_VIS = os.path.join(args.out_dir, 'removal_vis')

if not os.path.exists(BG_MASK_DIR): os.makedirs(BG_MASK_DIR)
if not os.path.exists(BG_THRESH_DIR): os.makedirs(BG_THRESH_DIR)
if not os.path.exists(RAW_SMALL): os.makedirs(RAW_SMALL)
if not os.path.exists(PEN_VIS): os.makedirs(PEN_VIS)
if not os.path.exists(REMOVAL_VIS): os.makedirs(REMOVAL_VIS)

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


    scale_thumbnail = 1/(MAG_BG_DET)
    thumbnail = region.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    thumbnail.save(os.path.join(RAW_SMALL, f'{basename}.png'))

    region = region.convert('RGB')
    res_dict = wsi_tissue_seg(region)

    # save histograms with thresholds
    fig, ax = plot_rgb_hist(res_dict['R'], res_dict['G'], res_dict['B'], res_dict['thr'])
    plt.show()
    fig.savefig(os.path.join(BG_THRESH_DIR, f'{basename}_thr.png'))
    plt.close(fig)

    # save marker removal effect results
    mask_pen = Image.fromarray(apply_mask(np.array(region), res_dict['mask_pen'], inv=False))
    mask_pen = mask_pen.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    mask_pen.save(os.path.join(PEN_VIS, f'{basename}_pen.png'))

    # save mask for grandQC
    np.save(os.path.join(BG_MASK_DIR, f'{basename}_mask.npy'), res_dict['mask'])
    mask = Image.fromarray((res_dict['mask']*255).astype(np.uint8))
    mask.save(os.path.join(BG_MASK_DIR, f'{basename}_mask.png'))

    # save mask visualisations on the tissue image
    region = np.array(region)
    vis_tissue = Image.fromarray(apply_mask(region, res_dict['mask'], inv=False))
    vis_tissue = vis_tissue.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    vis_tissue.save(os.path.join(REMOVAL_VIS, f'{basename}_thumbnail.png'))

    # save borders visualisation on the tissue image
    contours, _ = cv2.findContours(res_dict['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(region, contours, -1, (0, 0, 255), 2)
    region = Image.fromarray(region)
    region.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    region.save(os.path.join(REMOVAL_VIS, f'{basename}_contour.png'))

    log_slide = {}
    if os.path.exists(REPORT_FILE):
        pd.DataFrame(log_slide).to_csv(REPORT_FILE, mode="a", header=False, index=False)
    else:
        pd.DataFrame(log_slide).to_csv(REPORT_FILE, mode="w", header=True, index=False)



