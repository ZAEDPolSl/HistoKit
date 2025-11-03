import numpy as np
from PIL import Image
import cv2
from .artifacts import Artifact

def make_overlay(slide, wsi_heatmap_im, bg_mask, vis_size):
    """
    Overlay GrandQC artifact detection results on a downscaled WSI thumbnail.

    This function combines the original slide thumbnail with the artifact heatmap,
    highlights tissue boundaries, and ensures that background regions remain visible.

    Parameters
    ----------
    slide : ndarray of shape (H, W, 3)
        RGB image of the WSI region (full resolution or pre-downsampled).
    wsi_heatmap_im : PIL.Image.Image
        RGB mask image showing artifact detection results (e.g., colored by class).
    bg_mask : ndarray of shape (H, W)
        Binary mask indicating tissue regions (1 - tissue, 0 - background).
    vis_size : tuple of int (width, height)
        Desired size of the output thumbnail.

    Returns
    -------
    overlay : ndarray of shape (height, width, 3), dtype uint8
        RGB overlay image where artifact heatmap is created on the slide thumbnail.
        Tissue boundaries are drawn in blue, and background regions retain the original slide appearance.

    Notes
    -----
    - Uses `cv2.addWeighted` to blend the slide and heatmap with a fixed alpha (0.3 for slide, 0.7 for heatmap).

    Examples
    --------
    >>> overlay_image = make_overlay(slide_np, heatmap_image, tissue_mask, vis_size=(512, 512))
    >>> plt.imshow(overlay_image)
    >>> plt.show()

    """
    slide = Image.fromarray(slide)
    slide_reduced = slide.resize(vis_size, Image.Resampling.LANCZOS)
    heatmap_temp = wsi_heatmap_im.resize(slide_reduced.size, Image.Resampling.NEAREST)
    overlay = cv2.addWeighted(np.array(slide_reduced), 0.3, np.array(heatmap_temp), 0.7, 0)
    bg_mask = Image.fromarray(bg_mask)
    bg_mask = np.array(bg_mask.resize(vis_size, Image.Resampling.NEAREST))
    contours, _ = cv2.findContours(bg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heatmap_np = np.array(heatmap_temp)
    mask = np.all(heatmap_np == (128, 128, 128), axis=-1)
    overlay[mask] = np.array(slide_reduced)[mask]
    overlay = cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay

def make_artifacts_color_map(mask):
    """
    Convert a labeled mask into an RGB image using the Artifact Enum colors.

    Each label in the mask must correspond to an integer value of `Artifact`.

    Parameters
    ----------
    mask : numpy.ndarray of shape (H, W)
        Mask image with integer labels corresponding to `Artifact` Enum values.

    Returns
    -------
    rgb : numpy.ndarray of shape (H, W, 3), dtype uint8
        RGB image with each artifact mapped to its specified color.

    Notes
    -----
    - Any label in the mask not present in `Artifact` will be displayed in black.
    - Colors are defined in the `Artifact` Enum via `Artifact.color()`.

    Examples
    --------
    >>> import numpy as np
    >>> mask = np.array([[0, 1], [2, 3]])
    >>> rgb_image = make_artifacts_color_map(mask)
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(rgb_image)
    >>> plt.show()

    """
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)

    for artifact in Artifact:
        idx = mask == artifact.value
        color = artifact.color()
        r[idx] = color[0]
        g[idx] = color[1]
        b[idx] = color[2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb