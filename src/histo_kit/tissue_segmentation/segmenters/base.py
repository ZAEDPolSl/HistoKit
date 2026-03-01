from abc import ABC, abstractmethod
import numpy as np

from src.histo_kit.tissue_segmentation.segmenters.gamred.thresholding import get_thr_image


class TissueSegmenter(ABC):

    @abstractmethod
    def segment(self, region: np.ndarray):


        thr, R, G, B = get_thr_image(region)
