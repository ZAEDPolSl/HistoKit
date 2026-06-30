import numpy as np


def gaussian_window(h, w, sigma_factor=None):
    """
    Create a 2D Gaussian weighting window with values normalized to [0, 1].

    The window is symmetric and smoothly attenuates values towards the edges.
    It is used to blend overlapping patches in whole-slide inference
    to avoid visible stitching artifacts.

    Parameters
    ----------
    h : int
        Height of the window.
    w : int
        Width of the window.
    sigma_factor : float, optional
        Factor controlling the spread of the Gaussian.
        If ``None`` (default), the value is set to ``0.5 * max(h, w)``.

    Returns
    -------
    ndarray of shape (h, w)
        A 2D Gaussian window normalized such that its maximum value is 1.

    Notes
    -----
    - The Gaussian is computed independently in x and y dimensions and combined
      via an outer product.
    - Normalization ensures the central peak equals 1.

    Examples
    --------
    >>> win = gaussian_window(5, 5)
    >>> win.shape
    (5, 5)
    >>> win.max()
    1.0
    """
    if sigma_factor is None:
        sigma = 0.5 * max(h, w)
    else:
        sigma = sigma_factor * max(h, w)


    xv = np.linspace(-1, 1, w)
    yv = np.linspace(-1, 1, h)
    gx = np.exp(-0.5 * (xv / (sigma / max(h,w)))**2)
    gy = np.exp(-0.5 * (yv / (sigma / max(h,w)))**2)
    win = np.outer(gy, gx)

    win = win / win.max()
    return win

def constant_window(h, w):
    """
    Create a constant weighting window of ones.

    This function generates a 2D array filled with ones, which can be used as a
    uniform weighting window for blending patches without any attenuation.

    Parameters
    ----------
    h : int
        Height of the window.
    w : int
        Width of the window.

    Returns
    -------
    ndarray of shape (h, w)
        A 2D array filled with ones.

    Examples
    --------
    >>> win = constant_window(3, 3)
    >>> win.shape
    (3, 3)
    >>> np.all(win == 1)
    True
    """
    return np.ones((h, w), dtype=np.float32)

def get_weights(window_type, h, w, **kwargs):
    """
    Get a weighting window based on the specified type.

    This function serves as a factory to generate different types of weighting
    windows for blending patches in whole-slide inference. Supported types include
    'gaussian' for smooth attenuation and 'constant' for uniform weighting.

    Parameters
    ----------
    window_type : str
        Type of the window to create. Supported values are 'gaussian' and 'constant'.
    h : int
        Height of the window.
    w : int
        Width of the window.

    Returns
    -------
    ndarray of shape (h, w)
        The generated weighting window.

    Raises
    ------
    ValueError
        If an unsupported `window_type` is provided.

    Examples
    --------
    >>> win = get_weights('gaussian', 5, 5, sigma=1.0)
    >>> win.shape
    (5, 5)
    >>> win.max()
    1.0

    >>> win = get_weights('constant', 3, 3)
    >>> win.shape
    (3, 3)
    >>> np.all(win == 1)
    True
    """
    if window_type == 'gaussian':
        return gaussian_window(h, w, **kwargs)
    elif window_type == 'constant':
        return constant_window(h, w)
    else:
        raise ValueError(f"Unsupported window type: {window_type}. Supported types are 'gaussian' and 'constant'.")
    


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