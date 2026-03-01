import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import opening, closing, remove_small_objects, disk


def remove_pen(img, pen_color, thr_low, thr_high, thr_back, disk_radius):

    strel = disk(disk_radius)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    if pen_color == "black":
        mask = (
                (np.abs(r - g) <= thr_low) &
                (np.abs(g - b) <= thr_low)
        )

    elif pen_color == "green":
        mask = (
                (g > thr_high) &
                ((r - b) < thr_low) &
                (g - 0.5 * (r + b) > 0)
        )

    else:
        raise ValueError(f"Unsupported pen color: {pen_color}")

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