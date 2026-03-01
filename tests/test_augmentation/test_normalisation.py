import cv2
import staintools
import numpy as np
from skimage import exposure

def rgb2od(img:np.ndarray)->np.ndarray:
    mask = img == 0
    img[mask] = 1
    return np.maximum(-1*np.log(img/255), 1e-6)

def od2rgb(OD:np.ndarray)->np.ndarray:
    OD = np.maximum(OD, 1e-6)
    return (255*np.exp(-1*OD)).astype(np.uint8)


def enhance_contrast(img: np.ndarray, lp: int = 2, hp: int = 98) -> np.ndarray:
    img_enhanced = img.copy()
    percentiles = np.array(np.percentile(img_enhanced, (lp, hp)))
    p_low, p_high = percentiles[0], percentiles[1]
    if p_low >= p_high:
        p_low, p_high = np.min(img_enhanced), np.max(img_enhanced)
    if p_high > p_low:
        img_enhanced = exposure.rescale_intensity(
            img_enhanced,
            in_range=(p_low, p_high),
            out_range=(0.0, 255.0),
        )
    return img_enhanced.astype(np.uint8)


def get_luminosity(img:np.ndarray, thr: float, lp=2, hp=98)->np.ndarray:
    img = img.astype(np.uint8)
    img = enhance_contrast(img, lp=lp, hp=hp)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = img_lab[:,:,0]/255.0
    tis_mask = L<thr
    if tis_mask.sum() == 0:
        raise ValueError("Tissue mask is empty")
    return tis_mask


def get_stain_matrix(img: np.ndarray, luminosity_threshold: float = 0.8, angular_percentile: float = 99) -> np.ndarray:
    """Stain matrix estimation.

    Args:
        img (:class:`numpy.ndarray`):
            Input image used for stain matrix estimation.

    Returns:
        :class:`numpy.ndarray`:
            Estimated stain matrix.

    """
    img = img.astype("uint8")  # ensure input image is uint8

    # convert to OD and ignore background
    tissue_mask = get_luminosity(
        img,
        threshold=luminosity_threshold,
    ).reshape((-1,))
    img_od = rgb2od(img).reshape((-1, 3))
    img_od = img_od[tissue_mask]

    # eigenvectors of covariance in OD space (orthogonal as covariance symmetric)
    _, eigen_vectors = np.linalg.eigh(np.cov(img_od, rowvar=False))

    # the two principle eigenvectors
    eigen_vectors = eigen_vectors[:, [2, 1]]

    # make sure vectors are pointing the right way
    eigen_vectors = vectors_in_correct_direction(e_vectors=eigen_vectors)

    # project on this basis.
    proj = np.dot(img_od, eigen_vectors)

    # angular coordinates with respect to the principle, orthogonal eigenvectors
    phi = np.arctan2(proj[:, 1], proj[:, 0])

    # min and max angles
    min_phi = np.percentile(phi, 100 - angular_percentile)
    max_phi = np.percentile(phi, angular_percentile)

    # the two principle colors
    v1 = np.dot(eigen_vectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
    v2 = np.dot(eigen_vectors, np.array([np.cos(max_phi), np.sin(max_phi)]))
    he = h_and_e_in_right_order(v1, v2)

    return he / np.linalg.norm(he, axis=1)[:, None]

def vectors_in_correct_direction(e_vectors: np.ndarray) -> np.ndarray:
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1

    return e_vectors

def h_and_e_in_right_order(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    if v1[0] > v2[0]:
        return np.array([v1, v2])
    return np.array([v2, v1])


class MacenkoNormalizer:

@staticmethod
def get_concentrations(img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
    """Estimate concentration matrix given an image and stain matrix.

    Args:
        img (:class:`numpy.ndarray`):
            Input image.
        stain_matrix (:class:`numpy.ndarray`):
            Stain matrix for haematoxylin and eosin stains.

    Returns:
        numpy.ndarray:
            Stain concentrations of input image.

    """
    od = rgb2od(img).reshape((-1, 3))
    x, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
    return x.T

def fit(self, target: np.ndarray) -> None:

    self.stain_matrix_target = self.extractor.get_stain_matrix(target)
    self.target_concentrations = self.get_concentrations(
        target,
        self.stain_matrix_target,
    )
    self.maxC_target = np.percentile(
        self.target_concentrations,
        99,
        axis=0,
    ).reshape((1, 2))
    # useful to visualize.
    self.stain_matrix_target_RGB = od2rgb(self.stain_matrix_target)

def transform(self, img: np.ndarray) -> np.ndarray:
    """Transform an image.

    Args:
        img (:class:`numpy.ndarray` of type :class:`numpy.uint8`):
            RGB input source image.

    Returns:
        :class:`numpy.ndarray`:
            RGB stain normalized image.

    """
    stain_matrix_source = self.extractor.get_stain_matrix(img)
    source_concentrations = self.get_concentrations(img, stain_matrix_source)
    max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
    source_concentrations *= self.maxC_target / max_c_source
    trans = 255 * np.exp(
        -1 * np.dot(source_concentrations, self.stain_matrix_target),
    )

    # ensure between 0 and 255
    trans[trans > 255] = 255  # noqa: PLR2004
    trans[trans < 0] = 0

    return trans.reshape(img.shape).astype(np.uint8)

def test_reinhard():
    # Read data
    target = staintools.read_image("/mnt/data/Tmp/jmerta/HE/tests/test_patch_extraction/test_extract_wsi/1408_3840_1664_4096.png")
    to_transform = staintools.read_image("/mnt/data/Tmp/jmerta/HE/tests/test_patch_extraction/test_extract_wsi/1536_3584_1792_3840.png")

    # Standardize brightness (optional, can improve the tissue mask calculation)
    target = staintools.LuminosityStandardizer.standardize(target)
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

    # Stain normalize
    normalizer = staintools.StainNormalizer(method='reinhard')
    normalizer.fit(target)
    transformed = normalizer.transform(to_transform)
    transformed.save("reinhard_transformed.png")