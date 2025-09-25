import numpy as np
import matplotlib.pyplot as plt
from src.tissue_seg.find_thr import get_thr_image
from src.tissue_seg.utils import apply_mask, remove_pen, remove_gray_stains, get_strel_disk, remove_small_objects
from scipy import ndimage as ndi

def wsi_tissue_seg(region, fill_holes=False):
    img_np = np.array(region)

    # get thresholds for each channel (GaMRed or Otsu when threshold is too low)
    thr, R, G, B = get_thr_image(img_np, thr_min=0.7 * 255, verbose=True)

    # remove black pen
    mask_pen = remove_pen(img_np, "black", 0.7, 0, thr, 5)
    img_np = apply_mask(img_np, mask_pen, inv=True)

    # get regions above background
    mask = ~(((img_np[..., 0] > thr["R"]) & (img_np[..., 1] > thr["G"])) |
             ((img_np[..., 0] > thr["R"]) & (img_np[..., 2] > thr["B"])) |
             ((img_np[..., 1] > thr["G"]) & (img_np[..., 2] > thr["B"])))

    # remove gray stains with low Chroma component
    mask = remove_gray_stains(img_np, mask)

    # morphological operations to clean the mask
    SE = get_strel_disk(2)
    mask = ndi.binary_closing(mask, SE)
    mask = ndi.binary_opening(mask, SE)
    mask = remove_small_objects(mask)

    return {"mask": mask, "mask_pen": mask_pen, "R": R, "G": G, "B": B, "thr":thr}

def plot_rgb_hist(R, G, B, thr):
    bins = np.arange(len(R))

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    channels = [("R", R, "Red"),
                ("G", G, "Green"),
                ("B", B, "Blue")]

    for ax, (name, hist, color) in zip(axs, channels):
        ax.bar(bins, hist, color=color)
        ax.axvline(thr[name], color="black", linestyle="--", linewidth=1.5)
        ax.set_title(f"{name} channel histogram")

    plt.tight_layout()
    return fig, axs