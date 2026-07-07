import os
import time
from typing import Dict
import numpy as np
from skimage.filters import threshold_otsu
from scipy.signal import convolve2d

from ....slide.bbox import BBox
from ....slide.mask import SpatialMask
from ...base import Segmenter
from ...collectors.base import OutputKind
from ...utils import apply_mask
from ....savers.base import Saver
from ....slide.slide import Slide
from .config import HeThrConfig

class HeThrSegmenter(Segmenter):

    """Segment tissue regions in whole-slide images using the H&E Otsu Thresholding algorithm developed by  B.A. Schreiber et al.

    For details see the paper:
    B.A. Schreiber et al.,
    Rapid Artefact Removal and Tissue Segmentation in Haematoxylin and Eosin Stained Biopsies.
    Scientific Reports (https://doi.org/10.1038/s41598-023-50183-4), 2024.
    
    Parameters 
    ---------- 
    config : HeThrConfig 
        Configuration object controlling thresholding, magnifications, pen-mark removal, post-processing, visualization, output paths, and saver settings. 
    output_collector : optional 
        Object used to collect intermediate and visualization outputs. If ``None``, it is created from ``config.build_output_collector()``. 
    saver : optional 
        Object used to save segmentation results. If ``None``, a ``Saver`` is initialized using ``config.saver``. 
        
    Attributes 
    ---------- 
    config : HeThrConfig 
        Segmenter configuration. 
    output_collector : optional 
        Collector used for storing generated outputs such as thumbnails, overlays, and histograms. 
    saver : Saver 
        Saver used to persist segmentation results. 
    
    """

    def __init__(
        self,
        config: HeThrConfig,
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

        print(f"Segmenting tissue for the slide: {basename} using H&E Thresholding algorithm (B.A. Schreiber et al. 2024) ...") if verbose else None

        t0 = time.perf_counter()
        print(f"Reading slide at magnification {self.config.tissdet_mag}...") 
        # 1. read region at chosen magnification
        region_np = np.array(slide.read_region(mag = self.config.tissdet_mag))

        # 2. normalize image to the range [0, 1]
        region_np = region_np / 255.0

        # 3. split region to color channels RGB
        r = region_np[:, :, 0].astype(float)
        g = region_np[:, :, 1].astype(float)
        b = region_np[:, :, 2].astype(float)

        # 4. calculate differences and ReLU outcome
        r2g_mask = np.maximum(r - g, 0)
        b2g_mask = np.maximum(b - g, 0)

        # 5. calculate point-waise product
        tissue_heatmap = r2g_mask * b2g_mask

        print(f"Calculating Otsu threshold...")

        # 6. do Otsu thresholding in the new space
        thr = threshold_otsu(r2g_mask * b2g_mask, nbins=self.config.nbins)

        mask = tissue_heatmap > thr

        if self.config.blur_kernel_width != 0:
            blur_kernel = np.ones((self.config.blur_kernel_width, self.config.blur_kernel_width))
            mask = convolve2d(mask, blur_kernel, mode = "same")
            mask = mask > 0

        # 8. Post-processing steps
        for step in self.config.postprocess_steps:
            mask = step(mask)
        
        # 9. Collect outputs
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

        # 9. Split mask into regions and scale to save magnification
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
            "method": "HeThr",
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

        # 10. Save results (optional)
        if save:
            self.saver.save(
                os.path.join(self.config.out_dir, "mask_gamred"),
                basename,
                res_dict,
            )

        return res_dict