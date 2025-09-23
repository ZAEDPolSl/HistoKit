import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hsv
from skimage.morphology import disk, opening, closing


def remove_pen(img, pen_color, thr_low, thr_high, thr_back, radius):
    """
    Remove the pen from mask.
    :param img: RGB image to remove the pen.
    :param pen_color: color of the pen
    :param thr_low: lower threshold.
    :param thr_high: upper threshold.
    :param thr_back: dictionary of bg thresholds for each color channel
    :param radius: radius of disk used as structuring element
    :return:
    """

    # set structuring element for morphology
    SE = disk(radius)

    # choose thresholds based on color
    if pen_color == 'black':
        img_hsv = rgb2hsv(img)
        mask = img_hsv[:,:,2] < thr_low
    elif pen_color == 'red':
        pass
    elif pen_color == 'green':
        pass
    elif pen_color == 'blue':
        pass
    else:
        raise ValueError('Invalid pen color')

    mask = mask.astype(bool)

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    mask = mask & (~((R > thr_back["R"]) & (G > thr_back["G"])) |
                   ((R > thr_back["R"]) & (B > thr_back["B"])) |
                   ((G > thr_back["G"]) & (B > thr_back["B"])))

    if np.any(mask):
        mask = opening(mask, SE)
        mask = closing(mask, SE)
        mask = binary_fill_holes(mask)

    return mask