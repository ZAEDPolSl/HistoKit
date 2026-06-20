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

    def __init__(self, lum_thr=0.8, reg=0.1, max_iter=100):
        self.lum_thr = lum_thr
        self.reg = reg
        self.max_iter = max_iter

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
