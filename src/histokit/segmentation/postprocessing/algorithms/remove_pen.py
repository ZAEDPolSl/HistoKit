from scipy.ndimage import binary_fill_holes
from skimage.morphology import opening, closing, remove_small_objects, disk
from typing import Literal
import numpy as np

def remove_pen(
    img: np.ndarray,
    pen_color: Literal["black", "green"],
    thr_low: int | None = None,
    thr_high: int | None = None,
    thr_back: dict[str, int] | None = None,
    disk_radius: int | None = None,
) -> np.ndarray:
    """
    Remove pen markings from an image.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image.
    pen_color : {"black", "green"}
        Color of the pen to be removed.
    thr_low : int, optional
        Lower threshold for pen detection. If None, a color-specific
        default value is used (12 for both black and green).
    thr_high : int, optional
        Upper threshold for pen detection. If None, a color-specific
        default value is used (0 for black, 150 for green).
    thr_back : dict[str, int], optional
        RGB thresholds used to identify the background.
        If None, ``{"R": 0, "G": 0, "B": 0}`` is used (i.e., no background thresholding).
    disk_radius : int, optional
        Radius of the structuring element used in morphological
        operations. Defaults to 9.

    Returns
    -------
    np.ndarray
        Image with pen markings removed.

    Raises
    ------
    ValueError
        If `pen_color` is not one of {"black", "green"}.
    """
    defaults = {
        "black": (12, 0),
        "green": (12, 150),
    }

    try:
        default_low, default_high = defaults[pen_color]
    except KeyError:
        raise ValueError(f"Unsupported pen color: {pen_color}")

    thr_low = default_low if thr_low is None else thr_low
    thr_high = default_high if thr_high is None else thr_high
    disk_radius = 9 if disk_radius is None else disk_radius
    thr_back = {"R": 255, "G": 255, "B": 255} if thr_back is None else thr_back



    strel = disk(disk_radius)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    if pen_color == "black":
        mask = (
                (np.abs(r - g) <= thr_low) &
                (np.abs(g - b) <= thr_low)
        )

    else:  # green pen
        mask = (
                (g > thr_high) &
                ((r - b) < thr_low) &
                (g - 0.5 * (r + b) > 0)
        )

    # Remove background from mask
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    bg_mask = (
            ((r > thr_back["R"]) & (g > thr_back["G"])) |
            ((r > thr_back["R"]) & (b > thr_back["B"])) |
            ((g > thr_back["G"]) & (b > thr_back["B"]))
    )

    mask &= ~bg_mask

    # Perform morphological operations
    if np.sum(mask):
        mask = remove_small_objects(mask, min_size=disk_radius * 10)
        mask = closing(mask, strel)
        mask = opening(mask, strel)
        mask = binary_fill_holes(mask)

    # Find individual green pixels and add to mask
    if pen_color == "green":
        r, g, b = img[..., 0], img[..., 1], img[..., 2]

        green_mask = (
                (g > 250)
                & (r < 150)
                & (b < 100)
                & (g - 0.5 * (r + b) > 0)
        )

        mask |= green_mask

    return mask