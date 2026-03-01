import numpy as np
from PIL import Image
from .thresholding import get_thr_image
from ...pipeline.registry import register_segmenter
from ...postprocessing.algorithms.remove_gray_stains import remove_gray_stains
from ...postprocessing.algorithms.remove_pen import remove_pen
from ....mask.mask import Mask
from ....tissue_segmentation.segmenters.base import TissueSegmenter
from .config import GaMRedConfig
from ....utils.apply_mask import apply_mask

@register_segmenter("gamred")
class GaMRedSegmenter(TissueSegmenter):

    def __init__(self, config: GaMRedConfig):
        self.config = config

    def segment(self, slide):
        region_np = np.array(slide.read_region(mag = self.config.tissdet_mag))

        # get thresholds for each channel (GaMRed or Otsu when threshold is too low)
        thr, R, G, B = get_thr_image(region_np, thr_min=self.config.thr_min)

        mask_pen_black = remove_pen(region_np, "black", 12, 0, thr, 9)
        mask_pen_green = remove_pen(region_np, "green", 12, 150, thr, 9)
        mask_pen = mask_pen_black | mask_pen_green

        if np.any(mask_pen):
            region_np = apply_mask(region_np, mask_pen, inv=True)

        # get regions above background
        mask = ~(((region_np[..., 0] > thr["R"]) & (region_np[..., 1] > thr["G"])) |
                 ((region_np[..., 0] > thr["R"]) & (region_np[..., 2] > thr["B"])) |
                 ((region_np[..., 1] > thr["G"]) & (region_np[..., 2] > thr["B"])))

        # remove gray stains with low Chroma component
        mask_chroma = remove_gray_stains(region_np)
        mask = mask & mask_chroma

        for step in self.config.postprocess_steps:
            mask = step(mask, self.config)

        mask = Mask(mask = mask,
                    mag=self.config.tissdet_mag,
                    exclude_values=(0, ))
        
        mask = mask.return_rescaled(mag = self.config.save_mag)

        return {"mask": mask.mask_array,
                "bbox":mask.bbox_list,
                "R": R,
                "G": G,
                "B": B,
                "thr": thr,
                "mag_det": self.config.tissdet_mag,
                "mpp_det": slide.get_mpp_at_mag(self.config.tissdet_mag),
                "mag_save": self.config.save_mag,
                "mpp_save": mask.mpp}

