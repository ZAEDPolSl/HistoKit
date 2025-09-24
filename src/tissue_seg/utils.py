import numpy as np
from skimage.color import rgb2hsv, rgb2lab
from scipy import ndimage as ndi

def get_strel_disk(radius):
    """
    Generate a disk structuring element with given radius.
    There are small differences between matlab strel('disk') and
    this implementation. Function was tested for different
    radius values: 3, 4, 5, 7, 8, 9 - it gives the same result for
    3, 5 and 9, but for 4, 7 and 8 there are small differences
    in the corners of strel. That happens because matlab uses
    radial decomposition of disk, while this function does not.
    :param radius: disk radius
    :return: SE - structuring element
    """
    d = np.arange(-radius+1, radius)
    x, y = np.meshgrid(d, d)
    SE = (x**2+y**2)<radius**2
    return SE

def remove_gray_stains(img, mask=None):
    """
    Remove gray stains with low chroma component
    :param img: numpy array with the RGB image
    :param mask: precalculated mask from previous steps (None for no mask - all pixels would be taken into account)
    :return: mask for the image with gray stains removed
    """
    img_tmp = rgb2lab(img).astype(np.float128)
    tmp = np.sqrt(img_tmp[:,:,1]**2 + img_tmp[:,:,2]**2)
    return mask & (tmp>2) if mask is not None else tmp>2

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

def apply_mask(img, mask, inv):
    """
    Apply the mask to the image.
    :param img: numpy array with a RBG image
    :param mask: mask to apply to the image
    :param inv: invert mask or not
    :return: img - image with mask applied
    """
    if np.sum(mask) == 0:
        return img
    if inv:
        mask = 1-mask if inv else mask

    for c in range(img.shape[2]):
        tmp = img[:,:,c]*mask.astype(int)
        tmp[tmp==0] = 255
        img[:,:,c] = tmp
    return img

def remove_small_objects(mask):
    """
    Remove objects with small area (smaller than 5% of image area)
    :return: mask - mask with small objects removed
    """
    mask = mask
    return mask