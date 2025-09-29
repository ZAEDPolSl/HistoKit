import time
import numpy as np
import matplotlib.pyplot as plt
from src.tissue_seg.find_thr import get_thr_image
from src.tissue_seg.utils import apply_mask, remove_pen, remove_gray_stains, get_strel_disk, remove_small_objects
from scipy import ndimage as ndi

def wsi_tissue_seg(region, fill_holes=False, open_disk_r=2, close_disk_r=2):
    """
    Run tissue region segmentation with GaMRed algorithm (or Otsu when threshold is too low).
    :param region: WSI region for tissue detection
    :param fill_holes: Fill holes in mask or not
    :param open_disk_r: radius of the disk used during opening operation
    :param close_disk_r: radius of the disk used during closing operation
    :return: dictionary with keys: "mask" tissue mask with detected  tissue region (1 - mask, 0 - background)
            "mask_pen": mask with detected black pen area, "R","G", "B": red, green, blue channels pixel counts,
            "thr": dictionary with threshold values for each color channel
    """
    img_np = np.array(region)
    # get thresholds for each channel (GaMRed or Otsu when threshold is too low)
    thr, R, G, B = get_thr_image(img_np, thr_min=0.7 * 255, verbose=False)

    # remove black pen
    mask_pen = remove_pen(img_np, "black", 0.7, 0, thr, 5)
    img_np = apply_mask(img_np, mask_pen, inv=True)

    # get regions above background
    mask = ~(((img_np[..., 0] > thr["R"]) & (img_np[..., 1] > thr["G"])) |
             ((img_np[..., 0] > thr["R"]) & (img_np[..., 2] > thr["B"])) |
             ((img_np[..., 1] > thr["G"]) & (img_np[..., 2] > thr["B"])))

    # remove gray stains with low Chroma component
    mask = remove_gray_stains(img_np, mask)

    # fill holes in mask (if fill_holes==True)
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    # morphological operations to clean the mask
    SE_close = get_strel_disk(close_disk_r)
    SE_open = get_strel_disk(open_disk_r)
    mask = ndi.binary_closing(mask, SE_close)
    mask = ndi.binary_opening(mask, SE_open)

    # remove small regions
    s = time.time()
    mask = remove_small_objects(mask)

    return {"mask": mask, "mask_pen": mask_pen, "R": R, "G": G, "B": B, "thr":thr}

def plot_rgb_hist(R, G, B, thr):
    """
    Create histograms for each color channel and plot the threshold values.
    :param R: pixel counts for red color channel
    :param G: pixel counts for green color channel
    :param B: pixel counts for blue color channel
    :param thr: threshold values for each color channel (dictionary)
    :return: fig, ax - figure and axes objects with created plots
    """
    bins = np.arange(len(R))

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    channels = [("R", R, "Red"),
                ("G", G, "Green"),
                ("B", B, "Blue")]

    for ax, (name, hist, color) in zip(axs, channels):
        ax.bar(bins, hist, color=color, width=1)
        ax.set_yscale("log")
        ax.axvline(thr[name], color="Orange", linestyle="--", linewidth=2)
        ax.set_title(f"{name} channel histogram: thr = {thr[name]:.2f}")
    fig.tight_layout()
    return fig, axs