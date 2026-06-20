import os
import time
from typing import Dict
import numpy as np
from ...base import Segmenter
from ...collectors.base import OutputKind
from ...utils import apply_mask
from .thresholding import get_thr_image
from ...postprocessing.algorithms.remove_gray_stains import remove_gray_stains
from ...postprocessing.algorithms.remove_pen import remove_pen
from ....savers.base import Saver
from ....slide.mask_utils import rescale_mask, scale_mask_to_bbox, split_regions
from ....slide.slide import Slide
from .config import GaMRedConfig

class GaMRedSegmenter(Segmenter):

    def __init__(self, config: GaMRedConfig):
        self.config = config
        self.output_collector = config.build_output_collector()
        self.saver = Saver(self.config.saver)

    def segment(self, slide: Slide, basename: str = "slide", verbose: bool = False) -> Dict:

        print(f"Segmenting tissue for the slide: {basename} using GaMRed algorithm...") if verbose else None

        t0 = time.perf_counter()

        # 1. read region at chosen magnification
        region_np = np.array(slide.read_region(mag = self.config.tissdet_mag))

        # 2. get thresholds
        thr, R, G, B = get_thr_image(region_np, thr_min=self.config.thr_min)

        # 3. remove pen marks if needed
        mask_pen_green = np.zeros(region_np.shape[:2], dtype=np.uint8)
        mask_pen_black = np.zeros(region_np.shape[:2], dtype=np.uint8)

        if self.config.remove_green_pen:
            mask_pen_green = remove_pen(
                region_np,
                "green",
                self.config.thr_green_pen[0],
                self.config.thr_green_pen[1],
                thr,
                self.config.disk_radius_green_pen,
            )

        if self.config.remove_black_pen:
            mask_pen_black = remove_pen(
                region_np,
                "black",
                self.config.thr_black_pen[0],
                self.config.thr_black_pen[1],
                thr,
                self.config.disk_radius_black_pen,
            )

        mask_pen = mask_pen_black | mask_pen_green

        if np.any(mask_pen):
            region_np = apply_mask(region_np, mask_pen, inv=True)

        # 4. get regions above background
        mask = ~(((region_np[..., 0] > thr["R"]) & (region_np[..., 1] > thr["G"])) |
                 ((region_np[..., 0] > thr["R"]) & (region_np[..., 2] > thr["B"])) |
                 ((region_np[..., 1] > thr["G"]) & (region_np[..., 2] > thr["B"])))


        # 5. Remove gray stains with low Chroma component
        if self.config.remove_gray_stains:
            mask_chroma = remove_gray_stains(region_np)
            mask = mask & mask_chroma

        # 6. Post-processing steps
        for step in self.config.postprocess_steps:
            mask = step(mask)
        
        # 7. Collect outputs
        thumb = np.array(slide.read_region(mag=self.config.vis_mag))

        self._collect(
            name="thumbnail",
            step="input",
            kind=OutputKind.IMAGE,
            data=thumb,
            basename=basename,
        )

        self._collect(
            name="tissue_overlay",
            step="visualisation",
            kind=OutputKind.MASK,
            data=mask,
            basename=basename,
            image=thumb,
        )

        self._collect(
            name="histograms",
            step="thresholding",
            kind=OutputKind.HISTOGRAM,
            data={
                "thr": thr,
                "R": R,
                "G": G,
                "B": B,
            },
            basename=basename,
        )

        mask = mask.astype(np.uint8) * 255
        mask_rescaled = rescale_mask(mask, self.config.save_mag / self.config.tissdet_mag)
        mask_array, bbox_list = split_regions(mask_rescaled)


        elapsed = time.perf_counter() - t0

        res_dict = {
            "basename": basename,
            "method": "GaMRed",
            "type": "tissue_mask",

            "mask": mask_array,
            "bbox": bbox_list,

            "mag_det": self.config.tissdet_mag,
            "mag_save": self.config.save_mag,

            "mag_l0": slide.mag,
            "mpp_l0": slide.mpp,
            "level_dimensions_0": np.array(slide.level_dimensions[0]),

            "thr": thr,
            "config": self.config.to_hdf5_dict(),
            "time": elapsed
        }

        # 8. Save results (optional)
        self.saver.save(os.path.join(self.config.out_dir, "mask_gamred"), basename, res_dict)
        
        return res_dict