import os
import cv2
from PIL import Image
from math import ceil, floor
import numpy as np
from skimage import measure
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




def get_patch_grid(regions, patch_size=256, overlap=0.7):
    """
    Generate a grid of patch coordinates for one or multiple bounding regions.
    Patches are placed with a given overlap and returned as coordinate lists.

    Parameters
    ----------
    regions : list of tuple
        List of bounding boxes, where each bounding box is defined as
        ``(y_min, x_min, y_max, x_max)`` in pixel coordinates.
    patch_size : int, optional
        Size (height and width) of the extracted patches (default is ``256``).
    overlap : float, optional
        Fraction of overlap between adjacent patches, where ``0.0`` means no overlap
        and ``1.0`` means fully overlapping patches (default is ``0.9``).

    Returns
    -------
    dict of list
        A dictionary containing lists of patch coordinates:

        - ``"x_start"`` : list of int, starting x-coordinates
        - ``"y_start"`` : list of int, starting y-coordinates
        - ``"x_end"`` : list of int, ending x-coordinates
        - ``"y_end"`` : list of int, ending y-coordinates

    Notes
    -----
    The function may return coordinates outside the image boundaries.
    It is the caller's responsibility to handle cropping or padding.

    Examples
    --------
    >>> regions = [(100, 200, 500, 800)]
    >>> coords = get_patch_grid(regions, patch_size=256, overlap=0.5)
    """
    coords = {"x_start": [], "y_start": [], "x_end": [], "y_end": []}

    for bbox in regions:

        stride = max(int(round(patch_size * (1.0 - overlap))), 1)

        y_0 = bbox[0] - stride
        x_0 = bbox[1] - stride

        tis_h = bbox[2] - y_0  # y_max - y_min
        tis_w = bbox[3] - x_0  # x_max - x_min

        num_x = ceil((tis_w - patch_size) / stride) + 1 if tis_w > patch_size else 1
        num_y = ceil((tis_h - patch_size) / stride) + 1 if tis_h > patch_size else 1

        for ix in range(num_x):
            x = x_0 + ix * stride
            for iy in range(num_y):
                y = y_0 + iy * stride

                coords["x_start"].append(x)
                coords["y_start"].append(y)
                coords["x_end"].append(x + patch_size)
                coords["y_end"].append(y + patch_size)

    return coords



