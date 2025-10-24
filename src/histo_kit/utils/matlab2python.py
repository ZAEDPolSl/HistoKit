import numpy as np
from tifffile import tifffile


def list2cell(list_obj):
    """
    Convert a Python list to a NumPy array of objects.

    Parameters
    ----------
    list_obj : list
        List of objects to convert.

    Returns
    -------
    cell : ndarray of object, shape (len(list_obj),)
        NumPy array containing the same objects as in the input list.

    Notes
    -----
    - Each element in the list is preserved as-is in the array.
    - Useful for interoperability with code expecting NumPy arrays of objects.
    """
    cell = np.empty((len(list_obj), ), dtype=object)
    for i, v in enumerate(list_obj):
        cell[i]=v
    return cell


def get_wsi_ind_matlab(path):
    """
    Get indices of WSI image layers, following MATLAB's `iminfo` convention.

    Parameters
    ----------
    path : str
        Path to the SVS file containing the WSI image.

    Returns
    -------
    indices : list of int
        Indices of WSI image layers in MATLAB convention (starting from 1).

    Notes
    -----
    - The function converts layer indices to start from 1 to match MATLAB behavior.
    """
    ind = []
    with tifffile.TiffFile(path) as tif:
        for i, l in enumerate(tif.pages):
            t_width = l.tags.get("TileWidth")
            if t_width is not None:
                ind.append(i+1) #matlab indexing
    return ind

def get_strel_disk(radius):
    """
    Generate a disk-shaped structuring element (SE) with the given radius.

    There are small differences compared to MATLAB's `strel('disk')`.
    This implementation was tested for radius values: 3, 4, 5, 7, 8, 9.
    It gives the same result as MATLAB for 3, 5, and 9, but for 4, 7, and 8
    there are minor differences at the corners. The discrepancy arises because MATLAB
    uses a radial decomposition of the disk, while this function does not.

    Parameters
    ----------
    radius : int
        Radius of the disk.

    Returns
    -------
    SE : ndarray of bool, shape (2*radius-1, 2*radius-1)
        Structuring element representing the disk. True values correspond to
        pixels inside the disk.

    Notes
    -----
    Special case for `radius=2` is hard-coded for exact shape.
    """
    if radius == 2:
        return np.array([
    [False, False, True, False, False],
    [False, True, True, True, False],
    [True, True, True, True, True],
    [False, True, True, True, False],
    [False, False, True, False, False]
    ])
    d = np.arange(-radius+1, radius)
    x, y = np.meshgrid(d, d)
    SE = (x**2+y**2)<radius**2
    return SE