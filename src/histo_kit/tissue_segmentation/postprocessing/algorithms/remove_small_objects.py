import numpy as np
from skimage import measure, morphology
from skimage.measure import label
from .clustering import cluster_regions

def remove_small_objects(mask):
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

    mask_res = morphology.remove_small_objects(mask.astype(bool), min_size=thr_area, connectivity=2)
    return mask_res
