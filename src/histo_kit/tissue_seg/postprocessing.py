import numpy as np
from skimage.color import rgb2lab
from skimage import measure, morphology
from scipy import ndimage as ndi
from skimage.measure import label
from .find_thr import otsuthresh
from ..utils.matlab2python import get_strel_disk


def cluster_regions(data, max_iters=100, tol=1e-4):
    """
    Cluster regions areas using a single-threaded KMeans algorithm with deterministic initialization.

    Parameters
    ----------
    data : array-like, shape (n_samples,)
        1D data vector to be clustered.
    max_iters : int, optional
        Maximum number of iterations (default is 100).
    tol : float, optional
        Tolerance for convergence (default is 1e-4).

    Returns
    -------
    labels : ndarray, shape (n_samples,)
        Cluster labels assigned to each sample.
    centers : ndarray, shape (n_clusters,)
        Coordinates of cluster centers.
    """

    k=2
    data = np.array(data, dtype=float)
    centroids = np.linspace(data.min(), data.max(), k)

    for _ in range(max_iters):

        distances = np.abs(data[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)

        # Default matlab implementation - when cluster is empty, create a new cluster center by assigning
        # its centroid position to the furthest point of another clusters
        if len(set(labels)) < 2:
            empty_label = set(range(k)) - set(labels)
            idx_non_empty = np.argmax(distances[labels != empty_label])
            labels[idx_non_empty] = empty_label


        # Calculate a new centroid position by calculating the mean
        # of samples assigned to this cluster.
        new_centroids = np.array([
            data[labels == i].mean() for i in range(k)
        ])

        if np.all(np.abs(new_centroids - centroids) < tol):
            break

        centroids = new_centroids

    return labels, centroids



def remove_gray_stains(img, mask=None):
    """
    Remove gray stains from an RGB image based on low chroma component.

    Parameters
    ----------
    img : ndarray, shape (H, W, 3)
        Input RGB image as a NumPy array.
    mask : ndarray of bool, shape (H, W), optional
        Precalculated mask from previous steps. If None, all pixels are considered
        for processing. Default is None.

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
    img_tmp = rgb2lab(img).astype(np.float128)
    tmp = np.sqrt(img_tmp[:,:,1]**2 + img_tmp[:,:,2]**2)
    return mask & (tmp>2) if mask is not None else tmp>2

def remove_black_pen(img, thr_low, thr_back, radius):
    """
    Generate a binary mask to remove black pen markings from an RGB image.

    The function identifies dark pen marks on tissue region by thresholding
    the image in the LAB color space and refining the result using morphological
    operations. The output is a boolean mask that can be used to remove or inpaint pen marks.

    Parameters
    ----------
    img : ndarray
        Input RGB image as a NumPy array of shape (M, N, 3).
    thr_low : float
        Lower threshold applied to the Value (V) channel in LABspace to
        detect dark regions corresponding to pen marks.
    thr_back : dict
        Dictionary containing background thresholds for each color channel.
        Calculated during background detection. Expected keys are ``"R"``, ``"G"``,
        and ``"B"`` with float values, e.g. ``{"R": 0.6, "G": 0.6, "B": 0.6}``.
    radius : int
        Radius of the disk-shaped structuring element used in morphological
        opening and closing operations.

    Returns
    -------
    mask : ndarray of bool, shape (M, N)
        Binary mask where ``True`` indicates detected pen regions.

    Notes
    -----
    - The function uses a disk-shaped structuring element obtained from
      :func:`get_strel_disk` for morphological operations.
    - This function does not remove pen marks of other colors.

    Examples
    --------
    >>> thr_back = {"R": 0.6, "G": 0.6, "B": 0.6}
    >>> mask = remove_black_pen(image, thr_low=10, thr_back=thr_back, radius=3)
    >>> image_no_pen = image.copy()
    >>> image_no_pen[mask] = 0
    """
    # set structuring element for morphology
    SE = get_strel_disk(radius)

    # choose thresholds based on color
    img_tmp = rgb2lab(img)
    mask = img_tmp[:, :, 0] < thr_low

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

def remove_small_objects(mask, mode="otsu"):
    """
        Remove small objects from a binary mask based on area.

        Objects smaller than a calculated threshold (roughly 5% of tissue area)
        are removed.

        Parameters
        ----------
        mask : ndarray of bool, shape (H, W)
            Binary mask containing objects to be filtered.
        mode: string "otsu" or "kmeans"
            Choose "otsu" to calculate the threshold based on two-step Otsu thresholding using histogram created from object sizes.
            Choose "kmeans" to calculate the threshold using kmeans method with the deterministic initialization.
            Centroids are set as minimum and maximum value in the areas vector.

        Returns
        -------
        mask_out : ndarray of bool, shape (H, W)
            Mask with small objects removed.
    """
    thr_area_low = round(10 ** (0.45 * np.log10(np.sum(mask))))

    props = measure.regionprops(label(mask.astype(bool)))
    areas = np.array([p.area for p in props])
    area_tmp = areas[areas > thr_area_low]

    if len(area_tmp) < 2:
        thr_area = area_tmp[0]
        mask_res = morphology.remove_small_objects(mask.astype(bool), min_size=thr_area, connectivity=2)
        return mask_res

    if mode == "otsu":
        # first otsu (maximize inter-class variance between small and large regions)
        areas = area_tmp
        counts, bins = np.histogram(areas, bins = 'fd')

        thr_area,_ = otsuthresh(counts)
        thr_area = np.min(areas) + thr_area*(np.max(areas)-np.min(areas))

        area_tmp = areas[areas<thr_area]

        if len(area_tmp) > 2:
             # second otsu (on small regions)
             counts_low, bins_low = np.histogram(area_tmp, bins='fd')
             thr_area, _ = otsuthresh(counts_low)
             thr_area = np.min(area_tmp) + thr_area * (np.max(area_tmp) - np.min(area_tmp))
    elif mode == "kmeans":
        idx, centers = cluster_regions(np.log10(area_tmp))

        if centers[0] > centers[1]:
            thr_area = min(area_tmp[idx == 0]) - 1
        else:
            thr_area = min(area_tmp[idx == 1]) - 1

        if np.sum(areas > thr_area) < 1:
            idx, centers = cluster_regions(areas)
            if centers[0] > centers[1]:
                thr_area = min(areas[idx == 0]) - 1
            else:
                thr_area = min(areas[idx == 1]) - 1
    else:
        raise ValueError("Unknown mode for small objects removal. Choose one of: otsu, kmeans")

    mask_res = morphology.remove_small_objects(mask.astype(bool), min_size=thr_area, connectivity=2)
    return mask_res



