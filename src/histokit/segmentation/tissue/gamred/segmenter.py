import os
import time
from typing import Dict
from histokit.slide.bbox import BBox
from histokit.slide.mask import SpatialMask
import numpy as np
from ...base import Segmenter
from ...collectors.base import OutputKind
from ...utils import apply_mask
from .thresholding import get_thr_image
from ...postprocessing.algorithms.remove_gray_stains import remove_gray_stains
from ...postprocessing.algorithms.remove_pen import remove_pen
from ....savers.base import Saver
from ....slide.slide import Slide
from .config import GaMRedConfig

class GaMRedSegmenter(Segmenter):

    """Segment tissue regions in whole-slide images using the GaMRed algorithm. 

    The segmenter reads a slide at a configured detection magnification, estimates RGB background thresholds, using the GaMRed algorithm, 
    optionally removes pen marks (green and black) and gray stains, applies post-processing steps, and returns tissue masks together 
    with bounding boxes and metadata. Optionally, the results are saved. 
    
    Parameters 
    ---------- 
    config : GaMRedConfig 
        Configuration object controlling thresholding, magnifications, pen-mark removal, post-processing, visualization, output paths, and saver settings. 
    output_collector : optional 
        Object used to collect intermediate and visualization outputs. If ``None``, it is created from ``config.build_output_collector()``. 
    saver : optional 
        Object used to save segmentation results. If ``None``, a ``Saver`` is initialized using ``config.saver``. 
        
    Attributes 
    ---------- 
    config : GaMRedConfig 
        Segmenter configuration. 
    output_collector : optional 
        Collector used for storing generated outputs such as thumbnails, overlays, and histograms. 
    saver : Saver 
        Saver used to persist segmentation results. 
        
    Notes 
    ----- 
    The GaMRed workflow includes RGB threshold estimation, optional green and black pen removal, optional gray stain removal, configured post-processing, mask splitting, bounding-box normalization, and optional result saving. """

    def __init__(
        self,
        config: GaMRedConfig,
        output_collector=None,
        saver=None,
    ):
        self.config = config

        self.output_collector = (
            output_collector
            if output_collector is not None
            else config.build_output_collector()
        )

        self.saver = (
            saver
            if saver is not None
            else Saver(self.config.saver)
        )

    def segment(self, 
                slide: Slide, 
                basename: str = "slide", 
                verbose: bool = False, 
                save: bool = True) -> Dict:

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
            data=mask.copy(),
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

        # 8. Split mask into regions and scale to save magnification

        mask = SpatialMask(mask.astype(np.uint8) * 255, mag=self.config.tissdet_mag)
        regions = mask.split_regions()

        mask_array = []
        bbox_list = []

        for r in regions:
            r = r.scale(target_mag=self.config.save_mag)
            mask_array.append(r.data)
            bbox_list.append(r.bbox.numpy())

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

        # 9. Save results (optional)
        if save:
            self.saver.save(
                os.path.join(self.config.out_dir, "mask_gamred"),
                basename,
                res_dict,
            )

        return res_dict