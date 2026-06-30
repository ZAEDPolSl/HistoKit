import numpy as np
from skimage import measure, morphology
from skimage.measure import label
from .clustering import cluster_regions

def remove_small_regions(mask, thr_area=None):
    """
    Remove small connected components from a binary mask.

    The minimum object area can be provided explicitly. If it is not
    provided, an initial threshold is estimated from the image size and
    may be further adjusted using clustering of connected component areas.

    Parameters
    ----------
    mask : numpy.ndarray
        Input binary mask.
    thr_area : int or float, optional
        Minimum area threshold. Objects smaller than this value are
        removed. If ``None``, the threshold is estimated automatically as 10^0.45 times the total image area,
        and may be further refined using k-means clustering of component areas.

    Returns
    -------
    numpy.ndarray
        Binary mask with small objects removed.

    Notes
    -----
    Connected components are labeled from ``mask.astype(bool)``. When
    enough components are present, their areas are clustered in log-space with a k-means algorithm (k=2)
    to adaptively select the area threshold between small and large objects.

    """
    if thr_area is None:
        thr_area = round(10 ** (0.45 * np.log10(mask.shape[0] * mask.shape[1])))
    props = measure.regionprops(label(mask.astype(bool)))
    areas = np.array([p.area for p in props])
    area_tmp = areas[areas > thr_area]
    if len(area_tmp)>1:
        idx, centers = cluster_regions(np.log10(area_tmp))

        if centers[0] > centers[1]:
            thr_area = min(area_tmp[idx == 0]) - 1
        else:
            thr_area = min(area_tmp[idx == 1]) - 1

    if np.sum(areas>thr_area)<1:
        idx, centers = cluster_regions(np.log10(areas))
        if centers[0] > centers[1]:
            thr_area = min(areas[idx == 0]) - 1
        else:
            thr_area = min(areas[idx == 1]) - 1

    mask_res = morphology.remove_small_objects(mask.astype(bool), max_size=thr_area, connectivity=2)
    return mask_res
