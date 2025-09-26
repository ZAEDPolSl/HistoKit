import os
import argparse
import timeit
import cv2
import numpy as np
import glob
import scipy.io
import torch
from openslide import OpenSlide
import matplotlib.pyplot as plt
import PIL.Image as Image
from scipy.io import loadmat
from src.grand_qc.utils import slide_info, make_overlay
from src.grand_qc.wsi_process import slide_process_single
from src.tissue_seg.tissue_seg import wsi_tissue_seg, plot_rgb_hist
from src.tissue_seg.utils import apply_mask, get_wsi_ind_matlab
from grand_qc.config import config
import tifffile

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


data = loadmat("/mnt/data/Tmp/jmerta/HE/test_data/C3L-00008-26_mask_all.mat")
print(data)

MAG_BG_DET = 2.5 # magnification for tissue detection
WSI_DIR = args.wsi_dir
PATCH_SIZE_MODEL = 512 # patch size for grand QC
ENCODER_MODEL = 'timm-efficientnet-b0'
ENCODER_MODEL_WEIGHTS = 'imagenet'
BG_CLASS = 7 # background class
MPP_MODEL = 1 # mpp for grand qc model (mpp=1 corresponds to magnification 10x)
scale_thumbnail = 0.1

model_grandQC = torch.load(args.grandqc_model, map_location=args.device)

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

# get slides names
slides = glob.glob(os.path.join(WSI_DIR, '*.svs'))

# Process WSIs
print(f"Found {len(slides)} WSIs in {WSI_DIR} directory.\n")
for slide in slides:

    # dict for results
    res_dict = {
        'mask_all': 0,  # mask with detected tissue region
        'mask_artifacts': 0,  # mask with artifacts detected by grandQC for given region
        'ind_WSI': 0,  # indexes for WSI image layers
        'ratio': 0,  # ratio for each layer
        'scale_val': 0,  #
        'thr': 0,  # thresholds calculated for R, G, B color channels
        'bbox': 0  # bounding box of tissue region
    }

    basename = os.path.basename(slide).split('.')[0]
    res_dict['basename'] = basename
    res_dict['ind_WSI'] = get_wsi_ind_matlab(slide)

    # load slide
    slide = OpenSlide(slide)

    # get resizing ratio for each layer
    ratio = slide.level_downsamples
    res_dict['ratio'] = ratio


    # load image with 2.5 magnification (if it is not available - rescale it from the highest magnification)
    mag = float(slide.properties["openslide.objective-power"])  # get magnification of the 0 layer
    mag_layers = [round(mag / r, 2) for r in ratio]  # magnification of each layer
    mpp_slide = 10/mag_layers[0] # approximated slide mpp

    if MAG_BG_DET in mag_layers:
        mag_idx = mag_layers.index(MAG_BG_DET)
        w, h = slide.level_dimensions[mag_idx]
        region = slide.read_region((0, 0), mag_idx, (w, h))
        scale_val = ratio[mag_idx]
    else:
        mag_idx = 0  # get the highest magnification and rescale
        w, h = slide.level_dimensions[mag_idx]
        region = slide.read_region((0, 0), mag_idx, (w, h))
        scale_val = MAG_BG_DET / mag
        region = region.resize((int(w * scale_val), int(h * scale_val)), Image.BICUBIC)

    res_dict['ratio'] = ratio
    res_dict['scale_val'] = scale_val

    thumbnail = region.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    thumbnail.save(os.path.join(RAW_SMALL, f'{basename}.png'))

    region = region.convert('RGB')
    res_dict_tiss_det = wsi_tissue_seg(region, args.fill_holes, args.close_disk_r, args.open_disk_r)

    # save histograms with thresholds
    fig, ax = plot_rgb_hist(res_dict_tiss_det['R'], res_dict_tiss_det['G'], res_dict_tiss_det['B'], res_dict_tiss_det['thr'])
    plt.show()
    fig.savefig(os.path.join(BG_THRESH_DIR, f'{basename}_thr.png'))
    plt.close(fig)

    # save marker removal effect results
    mask_pen = Image.fromarray(apply_mask(np.array(region), res_dict_tiss_det['mask_pen'], inv=False))
    mask_pen = mask_pen.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    mask_pen.save(os.path.join(PEN_VIS, f'{basename}_pen.png'))

    # save mask for grandQC
    if "npy" in args.save_mask_formats:
        np.save(os.path.join(BG_MASK_DIR, f'{basename}_mask.npy'), res_dict_tiss_det['mask'])

    # save mask as color thumbnail
    mask = Image.fromarray((res_dict_tiss_det['mask']*255).astype(np.uint8))
    mask = mask.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.Resampling.NEAREST)
    mask.save(os.path.join(BG_MASK_DIR, f'{basename}_mask_small.png'))

    # save mask visualisations on the tissue image
    region = np.array(region)
    vis_tissue = Image.fromarray(apply_mask(region, res_dict_tiss_det['mask'], inv=False))
    vis_tissue = vis_tissue.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    vis_tissue.save(os.path.join(REMOVAL_VIS, f'{basename}_thumbnail.png'))

    # save borders visualisation on the tissue image
    contours, _ = cv2.findContours(res_dict_tiss_det['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_con = region.copy()
    cv2.drawContours(region_con, contours, -1, (0, 0, 255), 2)
    region_con = Image.fromarray(region_con)
    region_con.resize((int(w * scale_thumbnail), int(h * scale_thumbnail)), Image.BICUBIC)
    region_con.save(os.path.join(REMOVAL_VIS, f'{basename}_contour.png'))

    ############################################################################################################
    # RUN GRAND QC FOR ARTIFACTS DETECTION
    ############################################################################################################

    p_s, patch_n_w_l0, patch_n_h_l0, w_l0, h_l0, obj_power = slide_info(slide, PATCH_SIZE_MODEL, MPP_MODEL, mpp_slide)

    h, w = res_dict_tiss_det["mask"].shape
    tis_det = Image.fromarray(1-res_dict_tiss_det["mask"].astype(np.uint8))
    tis_det = np.array(tis_det.resize((int(w*4), int(h*4)), Image.Resampling.NEAREST))

    map_tiss, full_mask = slide_process_single(model_grandQC, tis_det, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                               PATCH_SIZE_MODEL, config.colors, ENCODER_MODEL,
                                               ENCODER_MODEL_WEIGHTS, args.device, BG_CLASS, MPP_MODEL, mpp_slide, w_l0, h_l0,
                                               (w, h))

    stop = timeit.default_timer()
    map_path = os.path.join(BG_MASK_DIR , basename + "_map_QC.png")
    map_tiss.save(map_path)
    mask_path = os.path.join(BG_MASK_DIR , basename + "_mask_qc.npy")
    full_mask = Image.fromarray(full_mask)
    full_mask = full_mask.resize((int(region.size[0]), int(region.size[1])) , Image.Resampling.NEAREST)
    np.save(mask_path, np.array(full_mask))

    if "mat" in args.save_mask_formats:
        mat_dict = {
            'mask_all': 0,  # mask with detected tissue region
            'mask_artifacts': 0,  # mask with artifacts detected by grandQC for given region
            'ind_WSI': 0,  # indexes for WSI image layers
            'ratio': 0,  # ratio for each layer
            'scale_val': 0,  #
            'thr': 0,  # thresholds calculated for R, G, B color channels
            'bbox': 0  # bounding box of tissue region
        }
        scipy.io.savemat(os.path.join(BG_MASK_DIR, f'{basename}_mask.mat'), mat_dict, do_compression=True)

    del full_mask

    overlay = make_overlay(region, map_tiss, args.overlay_factor)

    del map_tiss

    # Save overlaid image
    overlay_im = Image.fromarray(overlay)
    overlay_im_name = os.path.join(BG_MASK_DIR , basename + "_overlay_QC.jpg")
    overlay_im.save(overlay_im_name)

    del overlay



