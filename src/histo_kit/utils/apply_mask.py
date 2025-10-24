import numpy as np


def apply_mask(img, mask, inv):
    """
    Apply a binary mask to an RGB image.

    Parameters
    ----------
    img : ndarray, shape (H, W, 3)
        Input RGB image as a NumPy array.
    mask : ndarray of bool, shape (H, W)
        Binary mask to apply to the image.
    inv : bool
        If True, invert the mask before applying it.

    Returns
    -------
    img_out : ndarray, shape (H, W, 3)
        Image with the mask applied. Pixels outside the mask are set to 255.

    Notes
    -----
    - If the mask has no active pixels (all False), the original image is returned.
    - The mask is applied independently to each color channel.
    - Pixels corresponding to False in the mask are replaced by 255.
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