import numpy as np

def gaussian_window(h, w, sigma=None):
    if sigma is None:
        sigma = 0.5 * max(h, w)

    xv = np.linspace(-1, 1, w)
    yv = np.linspace(-1, 1, h)
    gx = np.exp(-0.5 * (xv / (sigma / max(h,w)))**2)
    gy = np.exp(-0.5 * (yv / (sigma / max(h,w)))**2)
    win = np.outer(gy, gx)

    win = win / win.max()
    return win

def to_tensor_x(x, **kwargs):
    """
    Convert an RGB image from a NumPy array to a PyTorch tensor.

    This function transposes the image from HWC (height, width, channels) format
    to CHW (channels, height, width) format and casts it to `float32`, preparing
    it for input into PyTorch models.

    Parameters
    ----------
    x : ndarray of shape (H, W, C)
        RGB image as a NumPy array.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    tensor : ndarray of dtype float32, shape (C, H, W)
        Transposed and type-cast tensor suitable for PyTorch.

    Notes
    -----
    - The function does not normalize pixel values (e.g., to [0,1]). This should
      be done separately if required by the model.
    - This is a lightweight function and does not create a true `torch.Tensor`.
      To convert to a PyTorch tensor, use `torch.from_numpy(to_tensor_x(x))`.

    Examples
    --------
    >>> x_tensor = to_tensor_x(image_np)
    >>> print(x_tensor.shape)  # (3, H, W)

    .. footbibliography::
    """
    return x.transpose(2, 0, 1).astype('float32')