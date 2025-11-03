import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from openslide import OpenSlide
from scipy import ndimage as ndi
from .find_thr import get_thr_image
from .postprocessing import remove_black_pen, get_strel_disk, remove_small_objects, \
    remove_gray_stains
from ..utils.apply_mask import apply_mask
from ..utils.file_utils import get_basename, save_rescaled
from ..utils.matlab2python import get_wsi_ind_matlab
from ..utils.wsi import load_wsi_mag


def wsi_tissue_seg(region, fill_holes=False, open_disk_r=2, close_disk_r=2, rem_small_obj=True):
    """
    Segment tissue regions in a whole-slide image (WSI) region using GaMRed or Otsu algorithms.

    This function performs tissue detection on a WSI region by first computing
    per-channel thresholds (GaMRed algorithm with fallback to Otsu if thresholds
    are too low). It removes black pen markings, eliminates background and gray
    stains, optionally fills holes, and applies morphological operations to
    clean the mask.

    Parameters
    ----------
    region : array_like
        WSI image region (RGB) for tissue detection.
    fill_holes : bool, optional
        If True, fill holes in the tissue mask using binary filling. Default is False.
    open_disk_r : int, optional
        Radius of the disk-shaped structuring element used for morphological opening. Default is 2.
    close_disk_r : int, optional
        Radius of the disk-shaped structuring element used for morphological closing. Default is 2.
    rem_small_obj: bool, optional
        If True, remove small objects around the tissue region. Default is True.

    Returns
    -------
    result : dict
        Dictionary containing segmentation results and intermediate data:
        - ``"mask"`` : ndarray, tissue mask (1 = tissue, 0 = background)
        - ``"mask_pen"`` : ndarray, mask for detected black pen regions
        - ``"R"``, ``"G"``, ``"B"`` : ndarray, histograms of Red, Green, and Blue channels
        - ``"thr"`` : dict, threshold values for each color channel

    Notes
    -----
    - Thresholds are computed using :func:`get_thr_image`.
    - Black pen regions are removed using :func:`remove_pen`.
    - Gray stains with low chroma are removed via :func:`remove_gray_stains`.
    - Morphological opening and closing help refine the mask and remove noise.
    - Small objects in the mask are removed to keep only significant tissue regions.
    - Input images should be RGB NumPy arrays with pixel values in range [0, 255].

    Examples
    --------
    >>> result = wsi_tissue_seg(region, fill_holes=True, open_disk_r=3, close_disk_r=3)
    >>> tissue_mask = result["mask"]
    >>> print("Red channel histogram:", result["R"])
    >>> print("Thresholds:", result["thr"])

    References
    ----------
    Method is described in \ :footcite:p:`bioimaging25`.

    .. footbibliography::
    """
    img_np = np.array(region)
    # get thresholds for each channel (GaMRed or Otsu when threshold is too low)
    thr, R, G, B = get_thr_image(img_np, thr_min=0.7 * 255, verbose=False)

    # remove black pen
    mask_pen = remove_black_pen(img_np, 10,  thr, 5)
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
    if rem_small_obj:
        mask = remove_small_objects(mask)

    return {"mask": mask, "mask_pen": mask_pen, "R": R, "G": G, "B": B, "thr":thr}

def plot_rgb_hist(R, G, B, thr):
    """
    Plot histograms for each RGB channel with threshold indicators.

    This function creates a stacked plot of histograms for the Red, Green,
    and Blue channels of an image and overlays vertical lines indicating
    threshold values for each channel.

    Parameters
    ----------
    R : ndarray of shape (256,)
        Histogram of pixel counts for the Red channel.
    G : ndarray of shape (256,)
        Histogram of pixel counts for the Green channel.
    B : ndarray of shape (256,)
        Histogram of pixel counts for the Blue channel.
    thr : dict
        Dictionary of threshold values for each color channel:
        - ``"R"`` : threshold for Red channel
        - ``"G"`` : threshold for Green channel
        - ``"B"`` : threshold for Blue channel

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plotted histograms.
    axs : ndarray of matplotlib.axes.Axes
        Array of axes objects corresponding to each color channel subplot.

    Notes
    -----
    - Histograms are plotted on a logarithmic y-scale to better visualize
      differences in pixel counts.
    - Vertical dashed lines represent the thresholds specified in `thr`.

    Examples
    --------
    >>> fig, axs = plot_rgb_hist(R, G, B, thr)
    >>> plt.show()
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


def segment_tissue(slide_file, args, paths_dict):

    # slide basename
    basename = get_basename(slide_file)

    # load slide
    slide = OpenSlide(slide_file)

    # rescale region
    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(slide, args.tissdet_mag, allow_upscaling=True)
    region = np.array(region)

    # size for visualisations
    w_l0, h_l0 = slide.level_dimensions[0]
    mag_l0 = float(slide.properties["openslide.objective-power"])
    scale_vis = args.vis_mag/mag_l0
    vis_size = (int(w_l0 * scale_vis), int(h_l0 * scale_vis))

    # free memory
    del slide

    # save scaled region thumbnail
    save_rescaled(region, vis_size, os.path.join(paths_dict["raw_small"], f'{basename}.tiff'), writer="tifffile")

    res_dict = wsi_tissue_seg(region, args.fill_holes, args.close_disk_r, args.open_disk_r, args.remove_small_objects)

    # save histograms with thresholds
    fig, ax = plot_rgb_hist(res_dict['R'], res_dict['G'], res_dict['B'], res_dict['thr'])
    fig.savefig(os.path.join(paths_dict["bg_thr_hist"], f'{basename}.png'))
    plt.close(fig)

    # save borders visualisation on the tissue image
    contours, _ = cv2.findContours(res_dict['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(region, contours, -1, (0, 0, 255), 2)
    save_rescaled(region, vis_size, os.path.join(paths_dict["bg_removal_contour_vis"], f'{basename}.tiff'), writer="tifffile")
    del region

    save_dict = {
            'basename': basename,  # tissue file basename (without .svs extension)
            'mask_bg': res_dict['mask'].astype(np.uint8),  # mask with detected tissue region
            'ind_WSI': get_wsi_ind_matlab(slide_file),  # indexes for WSI image layers (idx from 1)
            'ratio': ratio,  # ratio for each layer
            'scale_val': scale_val,  # scale factor of masks
            'mask_mag': args.tissdet_mag, # magnification of tissue detection
            'thr': res_dict['thr'],  # thresholds calculated for R, G, B color channels,
            'mpp': mpp_slide,
            'mag_l0': mag_l0
        }

    # save data to .mat file
    scipy.io.savemat(os.path.join(paths_dict["masks"], f'{basename}.mat'), save_dict, do_compression=True)

    return basename