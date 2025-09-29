import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy.io
import torch
from openslide import OpenSlide
import PIL.Image as Image
from skimage import measure
from src.grand_qc.utils import slide_info, make_overlay
from src.grand_qc.wsi_process import slide_process_single, make_artifacts_color_map
from src.tissue_seg.tissue_seg import wsi_tissue_seg, plot_rgb_hist
from src.tissue_seg.utils import apply_mask, get_wsi_ind_matlab, list2cell
from grand_qc.config import config
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

"""
Script for tissue region detection with multiple threads
"""

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='/mnt/data/Datasets/HE_data/Labaj_UCEC/SVS/05_2024/')
parser.add_argument('--out_dir', type=str, help='Output directory', default='../test_data/res9/')
parser.add_argument('--split_regions', type=bool, help='If there are multiple regions on the slide save each of them to a separate file.', default=True)
parser.add_argument('--fill_holes', type=bool, help='Fill holes in the tissue or not', default=False)
parser.add_argument('--close_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image closing', default=2)
parser.add_argument('--open_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image opening', default=2)
parser.add_argument('--save_mask_formats', nargs='+',help='File formats to save masks, choose at least one from: npy, mat.', choices=["npy", "mat"],default=["npy", "mat"])
parser.add_argument('--device', help='Device used for artifacts detection: cuda or cpu, cuda is not recommended for many threads.', choices=["cuda", "cpu"],default="cpu")
parser.add_argument('--overlay_factor', help='Factor used for creating image overlay', default=0.60, type=float)
parser.add_argument('--grandqc_model', help='Path to GrandQC model weights (model for 10x magnification is used by default).',default="grand_qc/models/GrandQC_MPP1.pth", type=str)
parser.add_argument('--workers', help="Number of workers used to process images in parallel.", default=10, type=int,choices=range(1, os.cpu_count() + 1))
args = parser.parse_args()

MAG_BG_DET = 2.5  # magnification for tissue detection
WSI_DIR = args.wsi_dir

PATCH_SIZE_MODEL = 512  # patch size for grand QC
ENCODER_MODEL = 'timm-efficientnet-b0'
ENCODER_MODEL_WEIGHTS = 'imagenet'
BG_CLASS = 7  # background class
MPP_MODEL = 1  # mpp for grand qc model (mpp=1 corresponds to magnification 10x)
scale_thumbnail = 0.25  # factor used to scale small thumbnails to show algorithms results (scaled from magnification for tissue detection)

# Create folders for results
BG_MASK_DIR = os.path.join(args.out_dir, 'masks')  # masks with detected tissues and grandQC results (saved as npy arrays, mat files or both)
BG_MASK_VIS_DIR = os.path.join(args.out_dir, 'bg_masks_vis')  # masks with detected tissue regions [small PNG thumbnails]
BG_THRESH_DIR = os.path.join(args.out_dir, 'bg_thr_hist')  # histograms with bg thresholds for tissue detection [PNG]
RAW_SMALL = os.path.join(args.out_dir, 'raw_small')  # tissue image [small PNG thumbnails]
PEN_VIS = os.path.join(args.out_dir, 'pen_vis')  # results of pen removal [small PNG thumbnails]
REMOVAL_VIS = os.path.join(args.out_dir, 'bg_removal_vis')  # results of bg removal [small PNG thumbnails]
REMOVAL_CONT_VIS = os.path.join(args.out_dir, 'bg_removal_contour_vis')  # results of bg removal with blue contours [small PNG thumbnails]
GRANDQC_MAP_VIS = os.path.join(args.out_dir,'grandqc_map_vis')  # results of artifacts detection with GrandQC (color maps) [small PNG thumbnails]
GRANDQC_OVERLAY_VIS = os.path.join(args.out_dir, 'grandqc_overlay_vis')  # results of artifacts detection with GrandQC (map overlay on tissue regions) [small PNG thumbnails]
REGION_GRANDQC_VIS = os.path.join(args.out_dir, 'grandqc_vis_region')  # results of artifacts detection with GrandQC for each region (color maps) [small PNG thumbnails]


if not os.path.exists(BG_MASK_DIR): os.makedirs(BG_MASK_DIR)
if not os.path.exists(BG_MASK_VIS_DIR): os.makedirs(BG_MASK_VIS_DIR)
if not os.path.exists(BG_THRESH_DIR): os.makedirs(BG_THRESH_DIR)
if not os.path.exists(RAW_SMALL): os.makedirs(RAW_SMALL)
if not os.path.exists(PEN_VIS): os.makedirs(PEN_VIS)
if not os.path.exists(REMOVAL_CONT_VIS): os.makedirs(REMOVAL_CONT_VIS)
if not os.path.exists(REMOVAL_VIS): os.makedirs(REMOVAL_VIS)
if not os.path.exists(GRANDQC_MAP_VIS): os.makedirs(GRANDQC_MAP_VIS)
if not os.path.exists(GRANDQC_OVERLAY_VIS): os.makedirs(GRANDQC_OVERLAY_VIS)
if not os.path.exists(REGION_GRANDQC_VIS) and args.split_regions: os.makedirs(REGION_GRANDQC_VIS)

# get slides names
slides = glob.glob(os.path.join(WSI_DIR, '*.svs'))

# Process WSIs
print(f"Found {len(slides)} WSIs in {WSI_DIR} directory. Starting processing with {args.workers} workers...")

def process_slides(slide_arr):
    error_slides = []
    error_msgs = []
    deltas = []
    processed = []
    for slide_file in slide_arr:
        try:
            start = time.time()
            process_single_slide(slide_file)
            delta = time.time() - start
            deltas.append(delta)
            processed.append(slide_file)
        except Exception as e:
            error_slides.append(slide_file)
            error_msgs.append(str(e))
            deltas.append(0)
    return error_slides, error_msgs, deltas, processed


def process_single_slide(slide_file):

    # slide basename
    basename = os.path.basename(slide_file).split('.')[0]

    # load slide
    slide = OpenSlide(slide_file)

    # get resizing ratio for each layer
    ratio = slide.level_downsamples

    # load image with 2.5 magnification (if it is not available - rescale it from the highest magnification)
    mag = float(slide.properties["openslide.objective-power"])  # get magnification of the 0 layer
    mag_layers = [round(mag / r, 2) for r in ratio]  # magnification of each layer
    mpp_slide = 10 / mag_layers[0]  # approximated slide mpp

    if MAG_BG_DET in mag_layers:
        mag_idx = mag_layers.index(MAG_BG_DET)
        w, h = slide.level_dimensions[mag_idx]
        region = slide.read_region((0, 0), mag_idx, (w, h))
        scale_val = ratio[mag_idx]
    else:
        mag_idx = 0  # get the highest magnification and rescale
        w0, h0 = slide.level_dimensions[mag_idx]
        region = slide.read_region((0, 0), mag_idx, (w0, h0))
        scale_val = MAG_BG_DET / mag
        region = region.resize((int(w0 * scale_val), int(h0 * scale_val)), Image.BICUBIC)
        w, h = region.size

    # size for visualisations
    vis_size = (int(w * scale_thumbnail), int(h * scale_thumbnail))

    # save scaled region thumbnail
    thumbnail = region.resize(vis_size, Image.BICUBIC)
    thumbnail.save(os.path.join(RAW_SMALL, f'{basename}.png'))
    region = region.convert('RGB')

    ############################################################################################################
    # RUN TISSUE REGION SEGMENTATION
    ############################################################################################################

    res_dict = wsi_tissue_seg(region, args.fill_holes, args.close_disk_r, args.open_disk_r)

    # save histograms with thresholds
    fig, ax = plot_rgb_hist(res_dict['R'], res_dict['G'], res_dict['B'], res_dict['thr'])
    fig.savefig(os.path.join(BG_THRESH_DIR, f'{basename}_thr.png'))
    plt.close(fig)

    # save marker removal effect results
    mask_pen = Image.fromarray(apply_mask(np.array(region), res_dict['mask_pen'], inv=False))
    mask_pen = mask_pen.resize(vis_size, Image.BICUBIC)
    mask_pen.save(os.path.join(PEN_VIS, f'{basename}_pen-small.png'))

    # save mask as color thumbnail
    mask = Image.fromarray((res_dict['mask'] * 255).astype(np.uint8))
    mask = mask.resize(vis_size, Image.Resampling.NEAREST)
    mask.save(os.path.join(BG_MASK_VIS_DIR, f'{basename}_mask-small.png'))

    # save mask visualisation on the tissue image
    region = np.array(region)
    vis_tissue = Image.fromarray(apply_mask(region.copy(), res_dict['mask'], inv=False))
    vis_tissue = vis_tissue.resize(vis_size, Image.BICUBIC)
    vis_tissue.save(os.path.join(REMOVAL_VIS, f'{basename}_tiss-det-small.png'))

    # save borders visualisation on the tissue image
    contours, _ = cv2.findContours(res_dict['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_con = region.copy()
    cv2.drawContours(region_con, contours, -1, (0, 0, 255), 2)
    region_con = Image.fromarray(region_con)
    region_con.resize(vis_size, Image.BICUBIC)
    region_con.save(os.path.join(REMOVAL_CONT_VIS, f'{basename}_contour-small.png'))

    ############################################################################################################
    # RUN GRAND QC FOR ARTIFACTS DETECTION
    ############################################################################################################
    model_grandQC = torch.load(args.grandqc_model, map_location=args.device)  # load grandQC model
    p_s, patch_n_w_l0, patch_n_h_l0, w_l0, h_l0, obj_power = slide_info(slide, PATCH_SIZE_MODEL, MPP_MODEL, mpp_slide)

    h, w = res_dict["mask"].shape
    tis_det = Image.fromarray(1 - res_dict["mask"].astype(np.uint8))
    tis_det = np.array(tis_det.resize((int(w * 4), int(h * 4)), Image.Resampling.NEAREST))

    map_tiss, full_mask, tis_det = slide_process_single(model_grandQC, tis_det, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                               PATCH_SIZE_MODEL, config.colors, ENCODER_MODEL,ENCODER_MODEL_WEIGHTS,
                                               args.device, BG_CLASS, MPP_MODEL, mpp_slide, w_l0, h_l0, vis_size)

    # save color grandQC artifacts map
    map_path = os.path.join(GRANDQC_MAP_VIS, basename + "_grandqc-small.png")
    map_tiss = map_tiss.resize(vis_size, Image.Resampling.NEAREST)
    map_tiss.save(map_path)

    # save region with map overlay
    overlay = make_overlay(region, map_tiss,tis_det, vis_size)
    overlay_im = Image.fromarray(overlay)
    overlay_im.save(os.path.join(GRANDQC_OVERLAY_VIS, basename + "_overlay-small.png"))

    del map_tiss
    del overlay

    full_mask = Image.fromarray(full_mask)
    full_mask = full_mask.resize((int(region.shape[1]), int(region.shape[0])), Image.Resampling.NEAREST)
    full_mask = np.array(full_mask)

    tis_det = Image.fromarray(tis_det)
    tis_det = np.array(tis_det.resize((int(region.shape[1]), int(region.shape[0])), Image.Resampling.NEAREST))
    res_dict["mask"] = tis_det

    if not args.split_regions:
        save_dict = {
            'basename': basename, # tissue file basename (without .svs extension)
            'mask_all': res_dict['mask'],  # mask with detected tissue region
            'mask_art': full_mask,  # mask with artifacts detected by grandQC for given region
            'ind_WSI': get_wsi_ind_matlab(slide_file),  # indexes for WSI image layers (idx from 1)
            'ratio': ratio,  # ratio for each layer
            'scale_val': scale_val,  # scale factor of masks
            'thr': res_dict['thr'],  # thresholds calculated for R, G, B color channels
        }

        if "mat" in args.save_mask_formats:
            scipy.io.savemat(os.path.join(BG_MASK_DIR, f'{basename}_mask.mat'), save_dict, do_compression=True)
        if "npy" in args.save_mask_formats:
            np.savez(os.path.join(BG_MASK_DIR, f'{basename}_mask.npz'), **save_dict)
    else:
        label_img = measure.label(res_dict['mask'])
        props = measure.regionprops(label_img)

        save_dict = {
            'basename': basename, # tissue file basename (without .svs extension)
            'mask_all': [],  # mask with detected tissue region
            'mask_art': [],  # mask with artifacts detected by grandQC for given region
            'ind_WSI': get_wsi_ind_matlab(slide_file),  # indexes for WSI image layers (idx from 1)
            'ratio': ratio,  # ratio for each layer
            'scale_val': scale_val,  # scale factor of masks
            'thr': res_dict['thr'],  # thresholds calculated for R, G, B color channels
            'tiss_stats': []  # bbox converted to matlab notation
        }

        for n, region in enumerate(props):
            region_mask_bg = region.image.astype(np.uint8)  # 0/1
            bbox = region.bbox
            region_mask_grandqc = full_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] * region_mask_bg

            save_dict['mask_all'].append(region_mask_bg.astype(bool))
            save_dict['mask_art'].append(region_mask_grandqc)
            save_dict['tiss_stats'].append([bbox[0] + 1, bbox[1] + 1, bbox[2] + 1, bbox[3] + 1])

            region_mask_grandqc = make_artifacts_color_map(region_mask_grandqc, config.colors)
            region_mask_grandqc = Image.fromarray(region_mask_grandqc)
            region_mask_grandqc.save(os.path.join(REGION_GRANDQC_VIS, f'{basename}_R{n + 1}.png'))

        save_dict['mask_all'] = list2cell(save_dict['mask_all'])
        save_dict['mask_art'] = list2cell(save_dict['mask_art'])

        if "mat" in args.save_mask_formats:
            scipy.io.savemat(os.path.join(BG_MASK_DIR, f'{basename}_mask_all.mat'), save_dict, do_compression=True)
        if "npy" in args.save_mask_formats:
            np.savez(os.path.join(BG_MASK_DIR, f'{basename}_mask_all.npz'), **save_dict)

    del full_mask

if __name__ == "__main__":
    time_start = time.time()
    log_file = os.path.join(args.out_dir, "error_files.txt")

    with ThreadPoolExecutor(args.workers) as executor:
        k, m = divmod(len(slides), args.workers)
        slides_arr = [slides[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(args.workers)]
        futures = {executor.submit(process_slides, s): s for s in slides_arr}

        with tqdm(total=len(slides)) as pbar:
            for fut in as_completed(futures):
                src = futures[fut]
                e_s, e_m, d, p = fut.result()
                pbar.update(len(src))
                print(f"\nProcessed {len(p)} slides:")
                print("Slide names:")
                for name in p:
                    print(name)

                if len(e_s)>0:
                    with open(log_file, 'a') as f:
                        print(f"There was an error during processing {len(e_s)} slides: ")
                        for s, m in zip(e_s, e_m):
                            print(f"Slide: {s} - error message: {m}")
                            f.write(f"Slide: {s} - error message: {m}\n")

    print("Finished in time: {:.2f} min".format((time.time() - time_start)/60))











