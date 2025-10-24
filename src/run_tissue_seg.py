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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

from src.histo_kit.grand_qc.artifact_detection import slide_info, slide_process_single
from src.histo_kit.grand_qc.visualisation import make_overlay, make_artifacts_color_map
from src.histo_kit.tissue_seg.bg_segmentation import wsi_tissue_seg, plot_rgb_hist
from src.histo_kit.utils.apply_mask import apply_mask
from src.histo_kit.utils.file_utils import create_folder, save_rescaled, get_basename
from src.histo_kit.utils.matlab2python import get_wsi_ind_matlab, list2cell
from src.histo_kit.utils.patches import load_wsi_mag

"""
Script for tissue region detection with multiple threads
"""

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='/mnt/data/Datasets/HE_data/Labaj_UCEC/SVS/05_2024/')
parser.add_argument('--out_dir', type=str, help='Output directory', default='../test_data/res12/')
parser.add_argument('--split_regions', type=bool, help='If there are multiple regions on the slide save each of them to a separate file.', default=True)
parser.add_argument('--fill_holes', type=bool, help='Fill holes in the tissue or not', default=False)
parser.add_argument('--close_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image closing', default=2)
parser.add_argument('--open_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image opening', default=2)
parser.add_argument('--save_mask_formats', nargs='+',help='File formats to save masks, choose at least one from: npy, mat.', choices=["npy", "mat"],default=["npy", "mat"])
parser.add_argument('--workers', help="Number of workers used to process images in parallel.", default=10, type=int,choices=range(1, os.cpu_count() + 1))
parser.add_argument('--grandqc_model', help='Path to GrandQC model weights (model for 10x magnification is used by default).',default="/mnt/data/Tmp/jmerta/HE/models/GrandQC_MPP1.pth", type=str)
parser.add_argument('--grandqc_mpp', help='MPP for grand qc model (mpp=1 corresponds to magnification 10x, mpp=2.0 - 5x, mpp=1.5 - 7.5x)',default=1.0, type=float)
parser.add_argument('--tissdet_mag', help='Magnification used for tissue detection',default=10, type=float)
parser.add_argument('--scale_thumbnail', help='factor used to scale small thumbnails to show algorithms results (scaled from magnification for tissue detection).',default=0.1, type=int)
parser.add_argument('--patch_size_model', help='Patch size for grand QC.',default=512, type=int)
parser.add_argument('--encoder_model', help='Name of a model used as encoder for GrandQC', default='timm-efficientnet-b0', type=str)
parser.add_argument('--encoder_model_weights', help='Name of weights used for encoder model in GrandQC', default='imagenet', type=str)
args = parser.parse_args()

MAG_MODEL = 10/args.grandqc_mpp
DEVICE = "cpu"

# Create folders for results
BG_MASK_DIR = create_folder(args.out_dir, 'masks')  # masks with detected tissues and grandQC results (saved as npy arrays, mat files or both)
BG_MASK_VIS_DIR = create_folder(args.out_dir, 'bg_masks_vis')  # masks with detected tissue regions [small PNG thumbnails]
BG_THRESH_DIR = create_folder(args.out_dir, 'bg_thr_hist')  # histograms with bg thresholds for tissue detection [PNG]
RAW_SMALL = create_folder(args.out_dir, 'raw_small')  # tissue image [small PNG thumbnails]
PEN_VIS = create_folder(args.out_dir, 'pen_vis')  # results of pen removal [small PNG thumbnails]
REMOVAL_VIS = create_folder(args.out_dir, 'bg_removal_vis')  # results of bg removal [small PNG thumbnails]
REMOVAL_CONT_VIS = create_folder(args.out_dir, 'bg_removal_contour_vis')  # results of bg removal with blue contours [small PNG thumbnails]
GRANDQC_MAP_VIS = create_folder(args.out_dir,'grandqc_map_vis')  # results of artifacts detection with GrandQC (color maps) [small PNG thumbnails]
GRANDQC_OVERLAY_VIS = create_folder(args.out_dir, 'grandqc_overlay_vis')  # results of artifacts detection with GrandQC (map overlay on tissue regions) [small PNG thumbnails]
REGION_GRANDQC_VIS = create_folder(args.out_dir, 'grandqc_vis_region')  # results of artifacts detection with GrandQC for each region (color maps) [small PNG thumbnails]

# get slides names
slides = glob.glob(os.path.join(args.wsi_dir, '*.svs'))
slides = slides[:10]
# Process WSIs
print(f"Found {len(slides)} WSIs in {args.wsi_dir} directory. Starting processing with {args.workers} workers...")

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
    basename = get_basename(slide_file)

    # load slide
    slide = OpenSlide(slide_file)

    # rescale region
    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(slide, args.tissdet_mag, allow_upscaling=True)
    w, h = region.size

    # size for visualisations
    vis_size = (int(w * args.scale_thumbnail), int(h * args.scale_thumbnail))

    # save scaled region thumbnail
    save_rescaled(region, vis_size, os.path.join(RAW_SMALL, f'{basename}.png'))

    region = region.convert('RGB')
    region = np.array(region)

    ############################################################################################################
    # RUN TISSUE REGION SEGMENTATION
    ############################################################################################################

    res_dict = wsi_tissue_seg(region, args.fill_holes, args.close_disk_r, args.open_disk_r)

    # save histograms with thresholds
    fig, ax = plot_rgb_hist(res_dict['R'], res_dict['G'], res_dict['B'], res_dict['thr'])
    fig.savefig(os.path.join(BG_THRESH_DIR, f'{basename}.png'))
    plt.close(fig)

    # save marker removal effect results
    mask_pen = Image.fromarray(apply_mask(np.array(region), res_dict['mask_pen'], inv=False))
    save_rescaled(mask_pen, vis_size, os.path.join(PEN_VIS, f'{basename}.png'))

    # save mask as a binary thumbnail
    save_rescaled(res_dict['mask'], vis_size, os.path.join(BG_MASK_VIS_DIR, f'{basename}.png'),
                  rescale_method=Image.Resampling.NEAREST, mode='1')

    # save mask visualisation on the tissue image
    vis_tissue = Image.fromarray(apply_mask(region.copy(), res_dict['mask'], inv=False))
    save_rescaled(vis_tissue, vis_size, os.path.join(REMOVAL_VIS, f'{basename}.png'))

    # save borders visualisation on the tissue image
    contours, _ = cv2.findContours(res_dict['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_con = region.copy()
    cv2.drawContours(region_con, contours, -1, (0, 0, 255), 2)
    save_rescaled(region_con, vis_size, os.path.join(REMOVAL_CONT_VIS, f'{basename}.png'))

    ############################################################################################################
    # RUN GRAND QC FOR ARTIFACTS DETECTION
    ############################################################################################################
    model_grandQC = torch.load(args.grandqc_model, map_location=args.device)  # load grandQC model
    p_s, patch_n_w_l0, patch_n_h_l0, w_l0, h_l0, obj_power = slide_info(slide, args.patch_size_model, args.grandqc_mpp,
                                                                        mpp_slide)

    h, w = res_dict["mask"].shape
    tis_det = Image.fromarray(1 - res_dict["mask"].astype(np.uint8))
    tis_det = np.array(tis_det.resize((int(w * MAG_MODEL / args.tissdet_mag), int(h * MAG_MODEL / args.tissdet_mag)),
                                      Image.Resampling.NEAREST))

    map_tis, full_mask, tis_det = slide_process_single(model_grandQC, tis_det, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                                       args.patch_size_model, args.encoder_model,
                                                       args.encoder_model_weights,
                                                       DEVICE, args.grandqc_mpp, mpp_slide, w_l0, h_l0, vis_size)

    # save color grandQC artifacts map
    map_tis = save_rescaled(map_tis, vis_size, os.path.join(GRANDQC_MAP_VIS, f'{basename}.png'),
                            rescale_method=Image.Resampling.NEAREST)

    # save region with map overlay
    overlay = make_overlay(region, map_tis, tis_det, vis_size)
    overlay_im = Image.fromarray(overlay)
    overlay_im.save(os.path.join(GRANDQC_OVERLAY_VIS, f'{basename}.png'))

    del map_tis
    del overlay

    full_mask = Image.fromarray(full_mask)
    full_mask = full_mask.resize((int(region.shape[1]), int(region.shape[0])), Image.Resampling.NEAREST)
    full_mask = np.array(full_mask)

    tis_det = Image.fromarray(tis_det)
    tis_det = np.array(tis_det.resize((int(region.shape[1]), int(region.shape[0])), Image.Resampling.NEAREST))
    res_dict["mask"] = tis_det

    if not args.split_regions:
        save_dict = {
            'basename': basename,  # tissue file basename (without .svs extension)
            'mask_all': res_dict['mask'],  # mask with detected tissue region
            'mask_art': full_mask,  # mask with artifacts detected by grandQC for given region
            'ind_WSI': get_wsi_ind_matlab(slide_file),  # indexes for WSI image layers (idx from 1)
            'ratio': ratio,  # ratio for each layer
            'scale_val': scale_val,  # scale factor of masks
            'thr': res_dict['thr'],  # thresholds calculated for R, G, B color channels
        }

        if "mat" in args.save_mask_formats:
            scipy.io.savemat(os.path.join(BG_MASK_DIR, f'{basename}.mat'), save_dict, do_compression=True)
        if "npy" in args.save_mask_formats:
            np.savez(os.path.join(BG_MASK_DIR, f'{basename}.npz'), **save_dict)
    else:
        label_img = measure.label(res_dict['mask'])
        props = measure.regionprops(label_img)

        save_dict = {
            'basename': basename,  # tissue file basename (without .svs extension)
            'mask_all': [],  # mask with detected tissue region
            'mask_art': [],  # mask with artifacts detected by grandQC for given region
            'ind_WSI': get_wsi_ind_matlab(slide_file),  # indexes for WSI image layers (idx from 1)
            'ratio': ratio,  # ratio for each layer
            'scale_val': scale_val,  # scale factor of masks
            'thr': res_dict['thr'],  # thresholds calculated for R, G, B color channels
            'tiss_stats': []
            # bbox converted to matlab notation in .mat files (indexing from 1) in .npz files (indexing from 0)
        }

        bbox_mat = []
        bbox_py = []

        for n, region in enumerate(props):
            region_mask_bg = region.image.astype(np.uint8)  # 0/1
            bbox = region.bbox
            region_mask_grandqc = full_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] * region_mask_bg

            save_dict['mask_all'].append(region_mask_bg.astype(bool))
            save_dict['mask_art'].append(region_mask_grandqc)

            bbox_mat.append([bbox[0] + 1, bbox[1] + 1, bbox[2] + 1, bbox[3] + 1])
            bbox_py.append([bbox[0], bbox[1], bbox[2], bbox[3]])

            region_mask_grandqc = make_artifacts_color_map(region_mask_grandqc)
            region_mask_grandqc = Image.fromarray(region_mask_grandqc)
            region_mask_grandqc.save(os.path.join(REGION_GRANDQC_VIS, f'{basename}_R{n + 1}.png'))

        save_dict['mask_all'] = list2cell(save_dict['mask_all'])
        save_dict['mask_art'] = list2cell(save_dict['mask_art'])

        if "mat" in args.save_mask_formats:
            save_dict["tiss_stats"] = bbox_mat
            scipy.io.savemat(os.path.join(BG_MASK_DIR, f'{basename}.mat'), save_dict, do_compression=True)
        if "npy" in args.save_mask_formats:
            save_dict["tiss_stats"] = bbox_py
            np.savez(os.path.join(BG_MASK_DIR, f'{basename}.npz'), **save_dict)

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











