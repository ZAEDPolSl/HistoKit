from math import ceil
from histokit.patch_extractors.datasets.conditions import ExcludeBackground
import numpy as np
from .base import PatchDataset
from ..patch_writer import PatchImageWriter

class GridExtractorDataset(PatchDataset):

    def __init__(self,
                 region: np.ndarray,
                 patch_size: int = 512,
                 pad_value: int = 255,
                 prep_fn=None,
                 aug_fn=None,
                 exclude_fn=ExcludeBackground(bg_value=255, threshold=1.0),
                 bbox_list=None,
                 patch_writer: PatchImageWriter = None,
                 overlap=0.75,
                 grid_offset=0.5):

        super().__init__(region,
                         patch_size,
                         pad_value,
                         prep_fn,
                         aug_fn,
                         exclude_fn,
                         bbox_list,
                         patch_writer)

        self.grid_offset = grid_offset
        self.overlap = overlap
        self.coords = self.get_coords()


    def get_coords(self):
        coords = {"x_start": [], "y_start": [], "x_end": [], "y_end": [], "bbox_idx": [], "bbox": []}

        for idx, bbox in enumerate(self.bbox_list):

            x_min, y_min, x_max, y_max = bbox
            stride = max(int(round(self.patch_size * (1.0 - self.overlap))), 1)

            y_0 = y_min - stride * self.grid_offset
            x_0 = x_min - stride * self.grid_offset

            tis_h = y_max - y_0  # y_max - y_min
            tis_w = x_max - x_0  # x_max - x_min

            num_x = ceil((tis_w - self.patch_size) / stride) + 1 if tis_w > self.patch_size else 1
            num_y = ceil((tis_h - self.patch_size) / stride) + 1 if tis_h > self.patch_size else 1

            for ix in range(num_x):
                x = x_0 + ix * stride
                for iy in range(num_y):
                    y = y_0 + iy * stride

                    coords["x_start"].append(x)
                    coords["y_start"].append(y)
                    coords["x_end"].append(x + self.patch_size)
                    coords["y_end"].append(y + self.patch_size)
                    coords["bbox_idx"].append(idx)
                    coords["bbox"].append(bbox)
        return coords


