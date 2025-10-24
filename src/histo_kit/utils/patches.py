import os
import cv2
from PIL import Image
from math import ceil, floor
import math
import numpy as np
from tqdm import tqdm
import matplotlib.colors as colors
from ..grand_qc.artifacts import Artifact

heatmap_colors = [
   (0.0, "#00008B"),
   (0.33, "#00FF00"),
   (0.66, "#FFFF00"),
    (1.0, "#FF0000"),
]

def patch_wsi(region, patch_size, save_folder, bg_percent, overlap=0, extract_type="valid"):
    """
    Divide a whole-slide image region into patches and save them.

    This function extracts square patches of a given size from an RGB image,
    optionally overlapping, and saves only those patches that meet a background
    pixel threshold. Background pixels are assumed to be white ([255, 255, 255]).

    Parameters
    ----------
    region : ndarray of shape (H, W, 3)
        RGB image region (masked or unmasked). White pixels are treated as background.
    patch_size : int
        Size of the square patches to extract (in pixels).
    save_folder : str
        Path to the folder where extracted patches will be saved. The folder
        is created if it does not exist.
    bg_percent : float
        Maximum allowable fraction of background pixels per patch. Patches with
        more background than this are rejected. Range [0, 1].
    overlap : float, optional
        Fraction of overlap between adjacent patches. Must be in [0, 1). Default is 0.
    extract_type : str, optional
        If `"valid"`, only extract fully contained patches. Otherwise, this
        specifies a NumPy padding mode (e.g., `"constant"`, `"reflect"`, `"symmetric"`)
        for partially overlapping patches. Default is `"valid"`.

    Returns
    -------
    num_correct : int
        Number of patches successfully extracted (below background threshold).
    num_rejected : int
        Number of patches rejected due to excessive background pixels.

    Notes
    -----
    - Patch filenames include the top-left coordinates in the format
      `"patch_x_y.png"`.
    - When `extract_type` is not `"valid"`, the image is padded as needed.

    Examples
    --------
    >>> num_correct, num_rejected = patch_wsi(region, patch_size=256, save_folder="patches", bg_percent=0.5, overlap=0.2)
    >>> print(f"Saved {num_correct} patches, rejected {num_rejected} patches.")
    """
    assert 0 <= overlap < 1.0, "overlap must be in range [0,1)."
    assert 0.0 <= bg_percent <= 1.0, "bg_percent must be in range [0,1]."
    assert region.ndim == 3 and region.shape[2] == 3, "region_rgb must be a RGB image"

    stride = max(int(round(patch_size * (1.0 - overlap))), 1)

    H, W = region.shape[:2]

    if extract_type == "valid":
        pad_mode = None
    else:
        pad_mode = extract_type

    if pad_mode is None:
        num_x = floor((W - patch_size) / stride) + 1 if W >= patch_size else 0
        num_y = floor((H - patch_size) / stride) + 1 if H >= patch_size else 0
    else:
        num_x = ceil((W - patch_size) / stride) + 1 if W > patch_size else 1
        num_y = ceil((H - patch_size) / stride) + 1 if H > patch_size else 1

    if pad_mode is not None:
        needed_w = (num_x - 1) * stride + patch_size
        needed_h = (num_y - 1) * stride + patch_size
        pad_right = max(0, needed_w - W)
        pad_bottom = max(0, needed_h - H)
        pad_left = 0
        pad_top = 0

        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        if pad_mode == "constant":
            region = np.pad(region, pad_spec, mode=pad_mode, constant_values=255)
        else:
            region = np.pad(region, pad_spec, mode=pad_mode)

    os.makedirs(save_folder, exist_ok=True)

    num_correct = 0
    num_rejected = 0
    with tqdm(total=num_x * num_y) as pbar:
        for ix in range(num_x):
            x = ix * stride
            for iy in range(num_y):
                y = iy * stride
                patch = region[y:y + patch_size, x:x + patch_size].copy()

                if np.sum(np.all(patch == 255, axis=2)) / (patch_size * patch_size) < bg_percent:
                    # add coordinates of the left upper corner to the patch name
                    Image.fromarray(patch).save(os.path.join(save_folder, f"patch_{x}_{y}.png"))
                    num_correct += 1
                else:
                    num_rejected += 1
                pbar.update(1)
    return num_correct, num_rejected



def load_wsi_mag(wsi, desired_mag, rescale_method = Image.LANCZOS, verbose = False, allow_upscaling = True):
    """
    Load and rescale a whole-slide image (WSI) to a desired magnification.

    This function reads the WSI at the closest available level to the desired
    magnification. If the exact magnification is unavailable, it rescales the
    highest-resolution level using the specified resampling method. Optionally,
    upscaling is allowed when the desired magnification is higher than the
    native WSI magnification.

    Parameters
    ----------
    wsi : OpenSlide object
        OpenSlide WSI object to load and rescale.
    desired_mag : float
        Desired slide magnification (e.g., 10, 20, 40).
    rescale_method : PIL.Image.Resampling or int, optional
        Resampling method used when resizing the image. Options include
        `Image.BICUBIC`, `Image.BILINEAR`, `Image.BOX`, `Image.HAMMING`,
        `Image.LANCZOS`, `Image.NEAREST`. Default is `Image.LANCZOS`.
    verbose : bool, optional
        If True, prints information about the rescaling process. Default is False.
    allow_upscaling : bool, optional
        If True, allows upscaling when the desired magnification is higher than
        the highest magnification available. Default is True.

    Returns
    -------
    region : PIL.Image.Image
        Rescaled WSI region at the desired magnification (converted to RGB).
    scale_val : float
        Scaling factor applied relative to the highest-resolution WSI level.
        For example, if the highest level is 40x and desired magnification is 10x,
        `scale_val = 40/10 = 4`.
    info : str
        Information message describing whether the desired magnification was
        available or if rescaling/upscaling was applied.
    mpp_slide : float
        Approximate microns-per-pixel (MPP) of the slide based on the highest magnification.
    ratio : list of float
        List of downsample ratios for each WSI level.

    Notes
    -----
    - If the desired magnification is available among the WSI levels, no rescaling
      is performed.
    - Rescaling is performed from the highest magnification level if the exact
      desired magnification is unavailable.
    - The function converts any RGBA images to RGB.

    Examples
    --------
    >>> region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, desired_mag=10)
    >>> print(info)
    >>> region.show()
    """
    
    ratio = wsi.level_downsamples
    mag_l0 = float(wsi.properties["openslide.objective-power"])
    mag_layers = [round(mag_l0/r, 2) for r in ratio]
    mpp_slide = 10 / mag_layers[0] # approximated slide mpp
    
    if desired_mag in mag_layers:
        info = "Desired magnification is available"
        mag_idx = mag_layers.index(desired_mag)
        w, h = wsi.level_dimensions[mag_idx]
        region = wsi.read_region((0, 0), mag_idx, (w, h))
        scale_val = ratio[mag_idx]
    else:
        if desired_mag > mag_l0:
            info = "Desired slide magnification is larger than available, image will be magnified from the highest magnification available."
            if not allow_upscaling:
                raise ValueError("The desired magnification is smaller than the highest magnification available. "
                                 "The parameter allow_upscaling is set to False, so the image will not be upscaled. "
                                 "If you want to upscale the image, set the parameter allow_upscaling to True. ")
        else:
            info = "Desired resolution is not available, image will be rescaled from the highest magnification available."
        mag_idx = 0  # get the highest magnification and rescale
        w0, h0 = wsi.level_dimensions[mag_idx]
        region = wsi.read_region((0, 0), mag_idx, (w0, h0))
        scale_val = desired_mag/ mag_l0
        region = region.resize((int(w0 * scale_val), int(h0 * scale_val)), rescale_method)

    # convert RGBA to RGB
    region = region.convert("RGB")

    if verbose:
        print(info)
        
    return region, scale_val, info, mpp_slide, ratio

def read_region(wsi, mask_file, region_idx, desired_mag, notation ="python", allow_list = (Artifact.NORM, Artifact.BG_MODEL), tol=1e-2, resampling_method = Image.Resampling.LANCZOS):
    """
    Read a masked region from a whole-slide image (WSI) and rescale it to a desired magnification.

    This function extracts a specified region from a WSI using bounding box information
    stored in a mask file. It applies artifact filtering, rescales the region to the
    desired magnification, and converts background pixels to white.

    Parameters
    ----------
    wsi : OpenSlide object
        OpenSlide WSI object from which to read the region.
    mask_file : dict-like
        Dictionary or NumPy file containing region bounding boxes, masks, and scaling information.
    region_idx : int
        Index of the region to read from the mask file.
    desired_mag : float
        Target magnification for the output region.
    notation : {'python', 'matlab'}, optional
        Specifies whether bounding boxes use Python (0-based) or MATLAB (1-based) indexing.
        Default is "python".
    allow_list : tuple of Artifact enums, optional
        Artifacts to allow in the mask. Only pixels labeled with these artifact types
        will be kept. Default is `(Artifact.NORM, Artifact.BG_MODEL)`.
    tol : float, optional
        Tolerance for the difference between the desired downsample ratio and the
        best available level. Default is 1e-2.
    resampling_method : PIL.Image.Resampling, optional
        Resampling method used when resizing regions (e.g., `Image.Resampling.LANCZOS`).
        Default is `Image.Resampling.LANCZOS`.

    Returns
    -------
    region_masked : ndarray of shape (H, W, 3)
        Masked and rescaled RGB region. Background pixels are set to white ([255, 255, 255]).

    Notes
    -----
    - Reads the WSI at the level closest to the desired magnification. If an exact
      level is not available, the region is rescaled using the specified resampling method.
    - Masks are resized to match the extracted region, and only allowed artifact regions
      are retained.
    - Pixels outside allowed regions are set to white for visualization.

    Examples
    --------
    >>> region = read_region(wsi, mask_file, region_idx=0, desired_mag=10)
    >>> plt.imshow(region)
    >>> plt.show()
    """

    # Load bounding box of the region at the magnification 2.5x
    bbox = np.array(mask_file["tiss_stats"][region_idx])

    # change matlab indexing to python indexing
    if notation == "matlab":
        bbox = bbox-1

    mag_l0 = float(wsi.properties["openslide.objective-power"])
    desired_ratio = mag_l0 / desired_mag
    scale_val = mask_file["scale_val"] / desired_ratio
    bbox = (bbox * scale_val).astype(int)

    w0, h0 = wsi.level_dimensions[0]
    des_w, des_h = int(w0 / desired_ratio), int(h0 / desired_ratio)

    level = wsi.get_best_level_for_downsample(desired_ratio)

    # load region defined by bbox
    if not math.isclose(float(wsi.level_downsamples[level]), desired_ratio, rel_tol=tol):
        # when desired magnification is not available - read the layer with the ratio that
        # is the nearest larger ratio to the desired ratio and resize
        region = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB")
        region = region.resize((des_w, des_h), resampling_method)
        region = np.array(region)[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    else:
        # when desired ratio is available read the correct wsi layer
        region = np.array(wsi.read_region((bbox[0], bbox[1]), level, (bbox[2] - bbox[0], bbox[3] - bbox[1])).convert("RGB"))

    # load mask with artifacts
    mask_art = mask_file["mask_art"][region_idx]
    mask = np.zeros_like(mask_art)

    # take regions from allow list
    for i in allow_list:
        mask[mask_art == i.value] = 1

    # Resize mask for desired resolution
    mask = np.array(Image.fromarray(mask).resize((region.shape[1], region.shape[0]), Image.Resampling.NEAREST))

    # get only areas defined by mask
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    region_masked = region * mask_rgb

    # turn bg pixels to white
    region_masked[np.all(region_masked == [0, 0, 0], axis=-1)] = [255, 255, 255]

    return region_masked


def merge_patches(patches_folder, attention_scores, scale_factor = 1, alpha=0.2):
    """
    Merge image patches into a full image and optionally overlay an attention heatmap.

    This function reconstructs a full tissue image from extracted patches. If
    attention scores are provided, it generates a heatmap overlay showing the
    attention values and blends it with the tissue image.

    Parameters
    ----------
    patches_folder : str
        Path to the folder containing extracted image patches. Patch filenames
        should encode their coordinates, e.g., 'patch_x_y.png'.
    attention_scores : dict or None
        Dictionary mapping patch filenames to attention scores. If None,
        no heatmap is generated.
    scale_factor : float, optional
        Scaling factor applied to the final overlay and attention map images.
        Default is 1 (no scaling).
    alpha : float, optional
        Weight for blending the attention heatmap with the tissue image.
        The final image is `overlay = (1-alpha) * tissue + alpha * heatmap`.
        Lower alpha makes the tissue more visible. Default is 0.2.

    Returns
    -------
    overlay : PIL.Image.Image
        Reconstructed tissue image with attention heatmap overlay, rescaled
        if `scale_factor` ≠ 1.
    attention_map_rgb : PIL.Image.Image
        RGB image of the attention heatmap alone, rescaled if `scale_factor` ≠ 1.
    attention_map : ndarray of shape (H, W)
        2D NumPy array of per-pixel attention scores. When patches overlap,
        scores are averaged per pixel.

    Notes
    -----
    - The function expects patches to have filenames encoding their top-left
      coordinates as `*_x_y.*`.
    - Overlapping patches are handled by averaging attention scores where they overlap.
    - Uses a colormap (`config.heatmap_colors`) to map attention values to RGB colors.
    - The final overlay uses OpenCV blending for alpha compositing.
    - Image resizing is done with nearest-neighbor interpolation.

    Examples
    --------
    >>> overlay, heatmap_rgb, attention_map = merge_patches("patches/", attention_scores, scale_factor=2, alpha=0.3)
    >>> overlay.show()
    >>> heatmap_rgb.show()
    >>> print("Attention map shape:", attention_map.shape)
    """
    patch_names = os.listdir(patches_folder)
    coords = {"x": [int(p.split("_")[1].split(".")[0]) for p in patch_names],
              "y": [int(p.split("_")[2].split(".")[0]) for p in patch_names]}

    max_x = max(coords["x"])
    max_y = max(coords["y"])

    sample_patch = Image.open(os.path.join(patches_folder, patch_names[0]))
    patch_size = sample_patch.size[0]
    width = max_x + patch_size
    height = max_y + patch_size

    cmap = colors.LinearSegmentedColormap.from_list("blue_green_yellow_red", heatmap_colors)

    final_im = np.zeros((width, height, 3), dtype=np.uint8)
    attention_map = np.zeros((width, height), dtype=np.float64)
    weights_map = np.zeros((width, height), dtype=np.uint8)

    for path in patch_names:
        patch = Image.open(os.path.join(patches_folder, path))
        patch = np.array(patch)

        x_min_p = int(path.split("_")[1].split(".")[0])
        y_min_p = int(path.split("_")[2].split(".")[0])
        x_max_p = x_min_p + patch_size
        y_max_p = y_min_p + patch_size

        final_im[x_min_p:x_max_p, y_min_p:y_max_p, :] = patch
        attention_map[x_min_p:x_max_p, y_min_p:y_max_p] += attention_scores[os.path.basename(path)]
        weights_map[x_min_p:x_max_p, y_min_p:y_max_p] += 1

    weights_map[weights_map == 0] = 1
    attention_map /= weights_map
    attention_map_rgba = cmap(attention_map)

    attention_map_rgb = (attention_map_rgba[:, :, 0:3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(np.array(final_im), 1 - alpha, np.array(attention_map_rgb), alpha, 0)
    overlay = Image.fromarray(overlay)

    overlay = overlay.resize((int(width * scale_factor), int(height * scale_factor)), Image.Resampling.NEAREST)
    attention_map_rgb = Image.fromarray(attention_map_rgb).resize((int(width * scale_factor), int(height * scale_factor)), Image.Resampling.NEAREST)

    return overlay, attention_map_rgb, attention_map