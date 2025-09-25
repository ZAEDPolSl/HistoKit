import os
import argparse
import timeit
import cv2
import numpy as np
import pandas as pd
import glob
import scipy.io
import torch
from openslide import OpenSlide
import matplotlib.pyplot as plt
import PIL.Image as Image
from src.grand_qc.utils import slide_info, make_overlay
from src.grand_qc.wsi_process import slide_process_single
from src.tissue_seg.tissue_seg import wsi_tissue_seg, plot_rgb_hist
from src.tissue_seg.utils import apply_mask
from grand_qc.config import config

"""
Script for tissue region detection 
Original matlab implementation is available in: github.com/WSI_TissueSeg
"""

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='../test_data/wsi/')
parser.add_argument('--out_dir', type=str, help='Output directory', default='../test_data/res/')
parser.add_argument('--fill_holes', type=bool, help='Fill holes in the tissue or not', default=True)
parser.add_argument('--close_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image closing', default=2)
parser.add_argument('--open_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image opening', default=2)
parser.add_argument('--save_mask_formats', nargs='+', help='File formats to save masks, choose at least one from: npy, mat.', choices=["npy", "mat"], default=["npy", "mat"])
parser.add_argument('--ask_for_overwrite', type=bool, help='Ask for overwriting the report file', default=False)
parser.add_argument('--device', help='Device used for artifacts detection: cuda or cpu', choices=["cuda", "cpu"], default="cuda")
parser.add_argument('--overlay_factor', help='Factor used for creating image overlay', default=0.60, type=float)
parser.add_argument('--grandqc_model', help='Path to GrandQC model weights (model for 10x magnification is used by default).', default="grand_qc/models/GrandQC_MPP1.pth", type=str)
parser.add_argument('--num_workers', help="Number of workers used to process images in parallel.", default=4, type=int, choices=range(1,os.cpu_count()+1))
args = parser.parse_args()

MAG_BG_DET = 2.5 # magnification for tissue detection
WSI_DIR = args.wsi_dir
M_P_S_MODEL = 512
ENCODER_MODEL = 'timm-efficientnet-b0'
ENCODER_MODEL_WEIGHTS = 'imagenet'
BACK_CLASS = 7
MPP_MODEL = 1

model_prim = torch.load(args.grandqc_model, map_location=args.device)

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
    mpp_slide = 10/mag_layers[0]

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
    res_dict = wsi_tissue_seg(region, args.fill_holes, args.close_disk_r, args.open_disk_r)

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
    if "npy" in args.save_mask_formats:
        np.save(os.path.join(BG_MASK_DIR, f'{basename}_mask.npy'), res_dict['mask'])
    if "mat" in args.save_mask_formats:
        scipy.io.savemat(os.path.join(BG_MASK_DIR, f'{basename}_mask.mat'), res_dict, do_compression=True)

    # save mask as color thumbnail
    mask = Image.fromarray((res_dict['mask']*255).astype(np.uint8))
    mask = mask.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.Resampling.NEAREST)
    mask.save(os.path.join(BG_MASK_DIR, f'{basename}_mask_small.png'))

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

    ############################################################################################################
    # RUN GRAND QC FOR ARTIFACTS DETECTION
    ############################################################################################################

    p_s, patch_n_w_l0, patch_n_h_l0, w_l0, h_l0, obj_power = slide_info(slide, M_P_S_MODEL, MPP_MODEL, mpp_slide)

    h, w = res_dict["mask"].shape
    tis_det = Image.fromarray(1-res_dict["mask"].astype(np.uint8))
    tis_det = np.array(tis_det.resize((int(w*4), int(h*4)), Image.Resampling.NEAREST))

    map_tiss, full_mask = slide_process_single(model_prim, tis_det, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                               M_P_S_MODEL, config.colors, ENCODER_MODEL,
                                               ENCODER_MODEL_WEIGHTS, args.device, BACK_CLASS, MPP_MODEL, mpp_slide, w_l0, h_l0,
                                               (w, h))

    stop = timeit.default_timer()
    map_path = os.path.join(BG_MASK_DIR , basename + "_map_QC.png")
    map_tiss.save(map_path)
    mask_path = os.path.join(BG_MASK_DIR , basename + "_mask_qc.npy")
    full_mask = Image.fromarray(full_mask)
    full_mask = full_mask.resize((int(region.size[0]), int(region.size[1])) , Image.Resampling.NEAREST)
    np.save(mask_path, np.array(full_mask))

    del full_mask

    map_tiss = Image.open(map_path)
    overlay = make_overlay(region, map_tiss, args.overlay_factor)

    del map_tiss

    # Save overlaid image
    overlay_im = Image.fromarray(overlay)
    overlay_im_name = os.path.join(BG_MASK_DIR , basename + "_overlay_QC.jpg")
    overlay_im.save(overlay_im_name)

    del overlay



