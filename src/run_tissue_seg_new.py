import os
import argparse
import glob
import time
import torch
from src.histo_kit.grand_qc.artifact_detection_optimized import process_single_optimized
from src.histo_kit.tissue_seg.bg_segmentation import segment_tissue
from src.histo_kit.utils.file_utils import create_folder, get_basename
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
"""
Script for tissue region and artifacts detection
"""

parser = argparse.ArgumentParser()

# Common settings
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='/mnt/data/Tmp/jmerta/test_svs/')
parser.add_argument('--out_dir', type=str, help='Output directory', default='/mnt/data/Tmp/jmerta/test_svs_res')
parser.add_argument('--scale_thumbnail', help='factor used to scale small thumbnails to show algorithms results (scaled from magnification for tissue detection).',default=0.1, type=int)
parser.add_argument('--split_regions', type=bool, help='If there are multiple regions on the slide save each of them to a separate file.', default=True)

# Settings for background detection with thresholding methods
parser.add_argument('--fill_holes', type=bool, help='Fill holes in the tissue or not', default=False)
parser.add_argument('--close_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image closing', default=2)
parser.add_argument('--open_disk_r', type=int, help='Radius for disk strel used during mask cleaning with image opening', default=2)
parser.add_argument('--tissdet_mag', help='Magnification used for tissue detection',default=10, type=float)
parser.add_argument('--workers', help='Number of workers used for background tissue detection.',default=5, type=int,choices=range(1, os.cpu_count() + 1))

# Settings for artifact detection with GrandQC
parser.add_argument('--device', help='Device used for artifacts detection: cuda or cpu', choices=["cuda", "cpu"],default="cuda")
parser.add_argument('--workers_per_slide', help='Number of workers used in Pytorch dataset during artifact segmentation.',default=5, type=int,choices=range(1, os.cpu_count() + 1))
parser.add_argument('--batch_size', help='Batch size used during artifact segmentation.', default=64, type=int)
parser.add_argument('--grandqc_model', help='Path to GrandQC model weights (model for 10x magnification is used by default).',default="/mnt/data/Tmp/jmerta/HE/models/GrandQC_MPP1.pth", type=str)
parser.add_argument('--grandqc_mpp', help='MPP for grand qc model (mpp=1 corresponds to magnification 10x, mpp=2.0 - 5x, mpp=1.5 - 7.5x)',default=1.0, type=float)
parser.add_argument('--patch_size_model', help='Patch size for grand QC.',default=512, type=int)
parser.add_argument('--save_mag', help='The magnification for final tissue regions.',default=2.5, type=float)
parser.add_argument('--encoder_model', help='Name of a model used as encoder for GrandQC', default='timm-efficientnet-b0', type=str)
parser.add_argument('--encoder_model_weights', help='Name of weights used for encoder model in GrandQC', default='imagenet', type=str)
parser.add_argument('--overlap', help='Overlap factor during image patching, set 0 for no overlap and 1 for full overlap', default=0.75, type=float)
parser.add_argument('--blending_mode', help='Method used to merge GrandQC predictions for overlapping patches. Use one of: gaussian (for weighted average with gaussian kernel), average (for weighted average)', default="gaussian", choices=["gaussian", "average"], type=str)
parser.add_argument('--blending_sigma', help='Sigma used for gaussian blending mode. When sigma is none it will be set as 0.5*patch_size. This parameter is not used for the average mode.', default=None, type=float)

def detect_bg_slides(slide_arr):
    error_slides = []
    error_msgs = []
    deltas = []
    processed = []
    for slide_file in slide_arr:
        try:
            start = time.time()
            segment_tissue(slide_file, args, paths_dict)
            delta = time.time() - start
            deltas.append(delta)
            processed.append(slide_file)
        except Exception as e:
            error_slides.append(slide_file)
            error_msgs.append(str(e))
            deltas.append(0)
    return error_slides, error_msgs, deltas, processed


# Create folders for results
if __name__ == "__main__":

    args = parser.parse_args()

    paths_dict = {"masks": create_folder(args.out_dir, 'masks'),
                  "masks_grandqc": create_folder(args.out_dir, 'masks_grandqc'),
                  "bg_masks_vis": create_folder(args.out_dir, 'bg_masks_vis'),
                  "bg_thr_hist": create_folder(args.out_dir, 'bg_thr_hist'),
                  "raw_small": create_folder(args.out_dir, 'raw_small'),
                  "pen_vis": create_folder(args.out_dir, 'pen_vis'),
                  "bg_removal_vis": create_folder(args.out_dir, 'bg_removal_vis'),
                  "bg_removal_contour_vis": create_folder(args.out_dir, 'bg_removal_contour_vis'),
                  "grandqc_map_vis": create_folder(args.out_dir, 'grandqc_map_vis'),
                  "grandqc_overlay_vis": create_folder(args.out_dir, 'grandqc_overlay_vis'),
                  "grandqc_vis_region": create_folder(args.out_dir, 'grandqc_vis_region'),
                  "grandqc_vis_weights": create_folder(args.out_dir, 'grandqc_vis_weights')}

    # get slides names
    slides = glob.glob(os.path.join(args.wsi_dir, '*.svs'))

    # calculate grandQC model magnification
    MAG_MODEL = 10/args.grandqc_mpp

    # Process WSIs
    print(f"Found {len(slides)} WSIs in {args.wsi_dir} directory. Using {args.workers} workers for tissue detection.\n")
    print("====== STEP 1 =======: Tissue detection.\n")
    time_start = time.time()

    error_file = os.path.join(args.out_dir, "tissdet_error.txt")
    log_file = os.path.join(args.out_dir, "tissdet_log.txt")

    # remove old log files
    for f in [error_file, log_file]:
        if os.path.exists(f):
            os.remove(f)


    # with ThreadPoolExecutor(args.workers) as executor:
    #     k, m = divmod(len(slides), args.workers)
    #     slides_arr = [slides[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(args.workers)]
    #     futures = {executor.submit(detect_bg_slides, s): s for s in slides_arr}
    #
    #     with tqdm(total=len(slides)) as pbar:
    #         for fut in as_completed(futures):
    #             src = futures[fut]
    #             e_s, e_m, d, proc = fut.result()
    #             pbar.update(len(src))
    #
    #             if len(proc) > 0:
    #                 with open(log_file, 'a') as f:
    #                     for p in proc:
    #                         f.write(f"{p}\n")
    #
    #             if len(e_s) > 0:
    #                 with open(error_file, 'a') as f:
    #                     print(f"There was an error during processing {len(e_s)} slides: ")
    #                     for s, m in zip(e_s, e_m):
    #                         print(f"Slide: {s} - error message: {m}")
    #                         f.write(f"Slide: {s} - error message: {m}\n")
    #
    # print("Finished STEP 1 (Tissue detection) - in time: {:.2f} min".format((time.time() - time_start) / 60))
    # print(f"STEP 2: Artifacts Segmentation with GrandQC. Using {args.device} for GrandQC.")
    #
    tis_det_files = glob.glob(paths_dict["masks"]+'/*.mat')
    folder_tis_det = paths_dict["masks"]

    print(f"Found {len(slides)} WSIs in {args.wsi_dir} directory and {len(tis_det_files)} corresponding h5 files with tissue masks in {folder_tis_det} directory.\n")
    print("====== STEP 2 =======: Artifacts detection with GrandQC.\n")
    print(f"Loading GrandQC model weights from {args.grandqc_model}...")

    model = torch.load(args.grandqc_model, map_location=args.device)

    time_start = time.time()

    error_file = os.path.join(args.out_dir, "grandqc_error.txt")
    log_file = os.path.join(args.out_dir, "grandqc_log.txt")

    # remove old log files
    for f in [error_file, log_file]:
        if os.path.exists(f):
            os.remove(f)


    for s_f in tqdm(slides, total=len(slides), desc="Processing slides"):
        try:
            basename = get_basename(s_f)
            tis_det = os.path.join(folder_tis_det, basename+".mat")
            process_single_optimized(
                s_f, tis_det, args.batch_size,
                args.workers_per_slide, args.device,
                model, paths_dict, args.scale_thumbnail, args.overlap, mag_model=MAG_MODEL,
                mode=args.blending_mode, sigma=args.blending_sigma, save_mag=args.save_mag
            )

            with open(log_file, 'a') as f:
                f.write(f"{basename}\n")
        except Exception as e:
            print(f"There was an error processing slide: {basename} - {str(e)}")
            with open(error_file, 'a') as f:
                f.write(f"{basename} - {str(e)}\n")









