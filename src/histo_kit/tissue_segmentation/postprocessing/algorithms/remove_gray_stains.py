import numpy as np
from skimage.color import rgb2lab


def remove_gray_stains(img):
    """
    Remove gray stains from an RGB image based on low chroma component.

    Parameters
    ----------
    img : ndarray, shape (H, W, 3)
        Input RGB image as a NumPy array.

    Returns
    -------
    mask_out : ndarray of bool, shape (H, W)
        Mask indicating pixels where gray stains have been removed.

    Notes
    -----
    The function converts the image to LAB color space and computes the chroma
    component as sqrt(a^2 + b^2). Pixels with chroma greater than 2 are considered
    non-gray. If a mask is provided, the output is the intersection of the mask
    and the chroma threshold.
    """
    img_tmp = rgb2lab(img)
    tmp = np.sqrt(img_tmp[:,:,1]**2 + img_tmp[:,:,2]**2)
    return tmp>2