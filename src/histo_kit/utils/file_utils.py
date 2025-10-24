import os
import numpy as np
from PIL import Image


def create_folder(parent, folder_name):
    """
    Create a folder within a parent directory if it does not already exist.

    Parameters
    ----------
    parent : str
        Path to the parent directory where the new folder should be created.
    folder_name : str
        Name of the folder to create.

    Returns
    -------
    str or Path
        The function creates the folder on disk and returns the path to the created folder.

    Notes
    -----
    If the folder already exists, it will not be recreated.
    """
    path = os.path.join(parent, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_rescaled(image, new_size, save_path, rescale_method=Image.LANCZOS, mode='RGB'):
    """
    Rescale and save an image to the specified path.

    Parameters
    ----------
    image : numpy.ndarray or PIL.Image.Image
        Input image to be rescaled. If a NumPy array is provided, it will be converted to a PIL image.
    new_size : tuple of int
        Target size of the rescaled image in the form (width, height).
    save_path : str or Path
        Path where the rescaled image will be saved.
    rescale_method : int, optional
        Resampling method used for resizing (default is ``Image.LANCZOS``, for other methods refer to PIL documentation).
    mode : str, optional
        Color mode to convert the image to (default is ``'RGB'``, for grayscale use ``'L'``, for binary use ``'1'`` for other options refer to the PIL documentation).

    Returns
    -------
    PIL.Image.Image
        The rescaled image object.

    Notes
    -----
    The function saves the rescaled image to disk and returns the resulting PIL image instance.
    """

    # Convert to PIL image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Rescale image
    image = image.resize(new_size, rescale_method)

    # Convert color mode
    if image.mode != mode:
        image = image.convert(mode)

    # Save the rescaled image
    image.save(save_path)

    return image

def get_basename(path):
    """
    Get the base name of a file without its extension.

    Parameters
    ----------
    path : str or Path
        Full path to the file.

    Returns
    -------
    str
        The base name of the file (filename without extension).

    Examples
    --------
    >>> get_basename("/home/user/image.v1.test.tiff")
    'image.v1.test'
    """
    return os.path.splitext(os.path.basename(path))[0]

