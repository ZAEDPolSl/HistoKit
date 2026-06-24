from .conditions import ExcludeBackground
from ...slide.bbox import BBox
import numpy as np
from .base import PatchDataset
from ..patch_writer import PatchImageWriter


class RandomExtractorDataset(PatchDataset):

    def __init__(
        self,
            region: np.ndarray,
            patch_size: int = 512,
            pad_value: int = 255,
            prep_fn=None,
            aug_fn=None,
            exclude_fn=ExcludeBackground(bg_value=255, threshold=1.0),
            bbox_list: list[BBox] | None = None,
            patch_writer: PatchImageWriter = None,
            patch_number=1000,
            seed=42,
    ):
        super().__init__(region,
                         patch_size,
                         pad_value,
                         prep_fn,
                         aug_fn,
                         exclude_fn,
                         bbox_list,
                         patch_writer)

        self.patch_number = patch_number
        self.seed = seed
        self.coords = self.get_coords()

    def get_coords(self):
        rng = np.random.default_rng(self.seed)

        coords = {
            "x_start": [],
            "y_start": [],
            "x_end": [],
            "y_end": [],
            "bbox_idx": [],
            "bbox": [],
        }

        for _ in range(self.patch_number):
            bbox_idx = int(rng.integers(0, len(self.bbox_list)))
            bbox = self.bbox_list[bbox_idx]

            if not isinstance(bbox, BBox):
                bbox = BBox(bbox)

            x_min, y_min, x_max, y_max = bbox.xyxy_int

            max_x_start = max(x_min, x_max - self.patch_size)
            max_y_start = max(y_min, y_max - self.patch_size)

            x_start = int(rng.integers(x_min, max_x_start + 1))
            y_start = int(rng.integers(y_min, max_y_start + 1))

            coords["x_start"].append(x_start)
            coords["y_start"].append(y_start)
            coords["x_end"].append(x_start + self.patch_size)
            coords["y_end"].append(y_start + self.patch_size)
            coords["bbox_idx"].append(bbox_idx)
            coords["bbox"].append(bbox.xywh_int)

        return coords