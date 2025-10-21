import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from ..tissue_seg.find_thr import get_thr_image
from ..tissue_seg.postprocessing import remove_black_pen, get_strel_disk, remove_small_objects, \
    remove_gray_stains
from ..wsi_utils.apply_mask import apply_mask


def wsi_tissue_seg(region, fill_holes=False, open_disk_r=2, close_disk_r=2):
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

    .. image:: example.png
        :alt: Example image
        :align: center
        :width: 400px



    References
    ----------
    Method is described in \ :footcite:p:`bioimaging25`.

    .. footbibliography::
    """
    img_np = np.array(region)
    # get thresholds for each channel (GaMRed or Otsu when threshold is too low)
    thr, R, G, B = get_thr_image(img_np, thr_min=0.7 * 255, verbose=False)

    # remove black pen
    mask_pen = remove_black_pen(img_np, 0.7,  thr, 5)
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