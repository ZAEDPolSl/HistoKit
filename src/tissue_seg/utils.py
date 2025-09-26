import numpy as np
from skimage.color import rgb2hsv, rgb2lab
from skimage import measure, morphology
from scipy import ndimage as ndi
from skimage.measure import label
from sklearn.cluster import KMeans
import tifffile


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
    :param mask: mask to remove objects
    :return: mask - mask with small objects removed
    """
    thr_area = round(10 ** (0.45 * np.log10(mask.shape[0] * mask.shape[1])))

    props = measure.regionprops(label(mask.astype(bool)))
    areas = np.array([p.area for p in props])
    area_tmp = areas[areas > thr_area]

    if len(area_tmp) < 2:
        thr_area = area_tmp[0]
        mask_res = morphology.remove_small_objects(mask.astype(bool), min_size=thr_area, connectivity=2)
        return mask_res

    kmeans = KMeans(n_clusters=2)
    idx = kmeans.fit_predict(np.log10(area_tmp).reshape(-1, 1))
    centers = kmeans.cluster_centers_.flatten()

    if centers[0] > centers[1]:
        thr_area = min(area_tmp[idx == 0]) - 1
    else:
        thr_area = min(area_tmp[idx == 1]) - 1


    if np.sum(areas > thr_area) < 1:
            kmeans = KMeans(n_clusters=2)
            idx = kmeans.fit_predict(areas.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            if centers[0] > centers[1]:
                thr_area = min(areas[idx == 0]) - 1
            else:
                thr_area = min(areas[idx == 1]) - 1

    mask_res = morphology.remove_small_objects(mask.astype(bool), min_size=thr_area, connectivity=2)
    return mask_res

def get_wsi_ind_matlab(path):
    ind = []
    with tifffile.TiffFile(path) as tif:
        for i, l in enumerate(tif.pages):
            t_width = l.tags.get("TileWidth")
            if t_width is not None:
                ind.append(i+1) #matlab indexing
    return ind

def list2cell(list):
    cell = np.empty((len(list), ), dtype=object)
    for i, v in enumerate(list):
        cell[i]=v
    return cell
