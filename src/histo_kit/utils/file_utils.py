import os

import h5py
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from tifffile import tifffile


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


def save_rescaled(image, new_size, save_path, rescale_method=Image.LANCZOS, mode='RGB', writer="PIL"):
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

    if writer == "PIL":
        # Convert color mode
        if image.mode != mode:
            image = image.convert(mode)

        # Save the rescaled image
        image.save(save_path)

    elif writer == "tifffile":
        image = np.array(image).astype(np.uint8)
        tifffile.imwrite(save_path, image, photometric='rgb', compression='lzw')

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


def save_mat_v73(file_path, save_dict):
    """
    Save dictionary as MATLAB -v7.3 HDF5 .mat file.
    Handles strings, lists, booleans, and numpy arrays.
    """
    with h5py.File(file_path, 'w') as f:
        for key, val in save_dict.items():
            # Convert booleans to uint8
            if isinstance(val, np.ndarray) and val.dtype == np.bool_:
                val = val.astype(np.uint8)
            # Convert lists to arrays
            elif isinstance(val, list):
                val = np.array(val)
            # Convert Python strings
            elif isinstance(val, str):
                val = np.array(val, dtype=h5py.string_dtype())
            # Scalars: wrap in array
            elif np.isscalar(val):
                val = np.array(val)
            # Ensure numpy array
            elif not isinstance(val, np.ndarray):
                raise TypeError(f"Cannot save key '{key}' of type {type(val)}")

            f.create_dataset(key, data=val)


