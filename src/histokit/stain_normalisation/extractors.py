from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import DictionaryLearning
from ..stain_normalisation.utils import (
    is_rgb_uint8,
    rgb2od,
    get_tissue_mask,
    normalize_matrix,
)

class BaseExtractor(ABC):

    @abstractmethod
    def get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        pass


class MacenkoExtractor(BaseExtractor):
    """Extract stain matrix using the algorith described byMacenko et al.

    This extractor estimates stain vectors from optical density values by
    projecting tissue pixels onto the plane spanned by the two principal
    eigenvectors of the optical density covariance matrix. Extreme angular
    directions in this plane are then used to estimate the stain vectors.

    The implementation is based on StainTools implementation.

    Parameters
    ----------
    ang_per : int, optional
        Angular percentile used to select the extreme projection angles.
        Default is ``99``.
    lum_thr : float, optional
        Luminosity threshold used to compute the tissue mask. Pixels brighter
        than this threshold are treated as background. Default is ``0.8``.

    Notes
    -----
    This implementation is based on the Vahadane stain extraction approach
    implemented in StainTools.

    References
    ----------
    .. [1] Macenko, M., Niethammer, M., Marron, J. S., Borland, D.,
       Woosley, J. T., Guan, X., Schmitt, C., & Thomas, N. E. (2009).
       A method for normalizing histology slides for quantitative analysis.
       In 2009 IEEE International Symposium on Biomedical Imaging:
       From Nano to Macro (pp. 1107-1110).
       https://doi.org/10.1109/ISBI.2009.5193250

    .. [2] Byfield, P. StainTools: Tools for tissue image stain normalization
       and augmentation in Python.
       https://github.com/Peter554/StainTools
    """

    def __init__(self, ang_per=99, lum_thr=0.8):
        self.ang_per = ang_per
        self.lum_thr = lum_thr



    def get_stain_matrix(self, img):
        assert is_rgb_uint8(img), "Image should be RGB uint8 np.ndarray"

        mask = get_tissue_mask(img, self.lum_thr)
        od = rgb2od(img).reshape((-1, 3))
        od = od[mask.reshape(-1)]

        _, v = np.linalg.eigh(np.cov(od, rowvar=False))
        v = v[:, [2, 1]]

        if v[0, 0] < 0:
            v[:, 0] *= -1
        if v[0, 1] < 0:
            v[:, 1] *= -1

        t = np.dot(od, v)
        phi = np.arctan2(t[:, 1], t[:, 0])

        min_phi = np.percentile(phi, 100 - self.ang_per)
        max_phi = np.percentile(phi, self.ang_per)

        v1 = np.dot(v, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(v, np.array([np.cos(max_phi), np.sin(max_phi)]))

        HE = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])

        return normalize_matrix(HE)


class VahadaneExtractor(BaseExtractor):
    """Extract stain matrix using the algorithm described by Vahadane et al.

    This extractor estimates the stain matrix for an RGB  image
    by applying dictionary learning to optical density values extracted from
    tissue pixels.

    Parameters
    ----------
    lum_thr : float, optional
        Luminosity threshold used to compute the tissue mask. Pixels brighter
        than this threshold are treated as background. Default is ``0.8``.
    reg : float, optional
        Regularization parameter used by ``DictionaryLearning``. Default is
        ``0.1``.
    max_iter : int, optional
        Maximum number of iterations for the dictionary learning algorithm.
        Default is ``100``.

    Notes
    -----
    This implementation is based on the Vahadane stain extraction approach
    implemented in StainTools.

    References
    ----------
    .. [1] Vahadane, A., Peng, T., Sethi, A., Albarqouni, S., Wang, L.,
       Baust, M., Steiger, K., Schlitter, A. M., Esposito, I., & Navab, N.
       (2016). Structure-Preserving Color Normalization and Sparse Stain
       Separation for Histological Images. IEEE Transactions on Medical
       Imaging, 35(8), 1962-1971. https://doi.org/10.1109/TMI.2016.2529665

    .. [2] Byfield, P. StainTools: Tools for tissue image stain normalization
       and augmentation in Python. https://github.com/Peter554/StainTools
    """

    def __init__(self, lum_thr=0.8, reg=0.1, max_iter=100):
        self.lum_thr = lum_thr
        self.reg = reg
        self.max_iter = max_iter

    """Estimate the stain matrix for an RGB image.

    The image is first converted to optical density space. A tissue mask is
    then used to select only tissue pixels, and dictionary learning with
    non-negative dictionary constraints is applied to estimate two stain
    vectors.

    Parameters
    ----------
    img : np.ndarray
        Input RGB image of type ``uint8`` with shape ``(H, W, 3)``.

    Returns
    -------
    np.ndarray
        Normalized stain matrix with shape ``(2, 3)``. Each row corresponds
        to one estimated stain vector.

    Raises
    ------
    AssertionError
        If ``img`` is not an RGB ``uint8`` NumPy array.
    """ 
    def get_stain_matrix(self, img):
        assert is_rgb_uint8(img), "Image should be RGB uint8 np.ndarray"

        mask = get_tissue_mask(img, self.lum_thr)

        od = rgb2od(img).reshape((-1, 3))
        od = od[mask.reshape(-1)]

        dict_learning = DictionaryLearning(
            n_components=2,
            alpha=self.reg,
            transform_alpha=self.reg,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_dict=True,
            verbose=False,
            max_iter=self.max_iter,
            transform_max_iter=1000,
        )

        dict_res = dict_learning.fit_transform(X=od.T).T

        if dict_res[0, 0] < dict_res[1, 0]:
            dict_res = dict_res[[1, 0], :]

        return normalize_matrix(dict_res)
