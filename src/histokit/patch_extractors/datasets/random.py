import numpy as np
from .base import PatchDataset
from .conditions import exclude_background
from ..patch_writer import PatchImageWriter


class RandomExtractorDataset(PatchDataset):

    def __init__(
        self,
            region: np.ndarray,
            patch_size: int = 512,
            pad_value: int = 255,
            prep_fn=None,
            aug_fn=None,
            exclude_fn=exclude_background(bg_value=255, threshold=1),
            bbox_list=None,
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
            bbox_idx = rng.integers(0, len(self.bbox_list))
            bbox = self.bbox_list[bbox_idx]

            x_min, y_min, x_max, y_max = map(int, bbox)

            max_x_start = max(x_min, x_max - self.patch_size)
            max_y_start = max(y_min, y_max - self.patch_size)

            x_start = rng.integers(x_min, max_x_start + 1)
            y_start = rng.integers(y_min, max_y_start + 1)

            coords["x_start"].append(int(x_start))
            coords["y_start"].append(int(y_start))
            coords["x_end"].append(int(x_start + self.patch_size))
            coords["y_end"].append(int(y_start + self.patch_size))
            coords["bbox_idx"].append(int(bbox_idx))
            coords["bbox"].append(bbox)

        print(coords)

        return coords