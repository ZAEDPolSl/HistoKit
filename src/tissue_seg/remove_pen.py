import numpy as np
from skimage.color import rgb2hsv
from scipy import ndimage as ndi

def get_strel_disk(radius):
    """
    Generate a disk structuring element with given radius.
    :param radius: disk radius
    :return: SE - structuring element
    """
    d = np.arange(-radius+1, radius)
    x, y = np.meshgrid(d, d)
    SE = (x**2+y**2)<radius**2
    return SE

def remove_gray_stains(img):
    mask = img
    return mask

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
    SE = get_strel_disk(radius)

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
        mask = ndi.binary_opening(mask, SE)
        mask = ndi.binary_closing(mask, SE)

    return mask