from math import ceil
from PIL import Image
from src.histokit.slide.slide import Slide
from .conditions import ExcludeBackground
from ...slide.bbox import BBox
import numpy as np
import torch

class GrandQCDataset:

    def __init__(self,
                 slide: Slide,
                 read_level: int = 0,
                 *,
                 target_mpp: float | None = None,
                 patch_size: int = 512,
                 pad_value: int = 255,
                 prep_fn=None,
                 aug_fn=None,
                 exclude_fn=ExcludeBackground(bg_value=255, threshold=1.0),
                 bbox_list: list[BBox] | None = None,
                 overlap=0.75,
                 grid_offset=0.5):

        self.grid_offset = grid_offset
        self.overlap = overlap
        self.coords = self.get_coords()
        self.slide = slide
        self.read_level = read_level
        self.target_mpp = target_mpp
        self.bbox_list = bbox_list
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.prep_fn = prep_fn
        self.aug_fn = aug_fn
        self.exclude_fn = exclude_fn
        self.patch_size_level = self.get_patch_size()


    def get_patch_size(self):
        mpp_level = self.slide.level_mpp[self.read_level]
        patch_size_level = round(self.patch_size * self.target_mpp / mpp_level)
        return patch_size_level


    def get_coords(self):
        coords = {"x_start": [], "y_start": [], "x_end": [], "y_end": [], "bbox_idx": [], "bbox": []}

        for idx, bbox in enumerate(self.bbox_list):

            # rescale bbox to desired level coordinates
            bbox = bbox.scale(target_mag=self.slide.level_mag[self.read_level])

            # calculate coordinates
            x_min, y_min, x_max, y_max = bbox.xyxy_int
            bbox_xywh = bbox.xywh_int

            stride = max(int(round(self.patch_size_level * (1.0 - self.overlap))), 1)

            y_0 = y_min - stride * self.grid_offset
            x_0 = x_min - stride * self.grid_offset

            y_0 = int(round(y_0))
            x_0 = int(round(x_0))

            tis_h = y_max - y_0
            tis_w = x_max - x_0

            num_x = (
                ceil((tis_w - self.patch_size_level) / stride) + 1
                if tis_w > self.patch_size_level
                else 1
            )

            num_y = (
                ceil((tis_h - self.patch_size_level) / stride) + 1
                if tis_h > self.patch_size_level
                else 1
            )

            for ix in range(num_x):
                x = x_0 + ix * stride

                for iy in range(num_y):
                    y = y_0 + iy * stride

                    coords["x_start"].append(x)
                    coords["y_start"].append(y)
                    coords["x_end"].append(x + self.patch_size_level)
                    coords["y_end"].append(y + self.patch_size_level)
                    coords["bbox_idx"].append(idx)
                    coords["bbox"].append(bbox_xywh)

        return coords

    def _crop_with_padding(self, x_start, y_start, x_end, y_end):
            
            # Crop patch with padding if necessary
            h, w = self.slide.level_dimensions[self.read_level]
    
            sx0 = max(0, x_start)
            sy0 = max(0, y_start)
            sx1 = min(w, x_end)
            sy1 = min(h, y_end)

            sh = sy1 - sy0
            sw = sx1 - sx0
    
            patch = np.array(self.slide.read_region(bbox = [sy0, sx0, sh, sw], level=self.read_level))

            if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
    
                padded = np.full(
                    (self.patch_size, self.patch_size, patch.shape[2]),
                    self.pad_value,
                    dtype=patch.dtype,
                )
    
                paste_x = max(0, -x_start)
                paste_y = max(0, -y_start)

                h_copy = min(patch.shape[0], self.patch_size - paste_y)
                w_copy = min(patch.shape[1], self.patch_size - paste_x)

                if h_copy > 0 and w_copy > 0:
                    padded[
                        paste_y:paste_y + h_copy,
                        paste_x:paste_x + w_copy,
                    ] = patch[:h_copy, :w_copy]

                    patch = padded

            # rescale path to the desired magnification/mpp
            patch_rescaled = Image.fromarray(patch).resize((self.patch_size, self.patch_size), resample=Image.LANCZOS)

            return patch_rescaled

    @staticmethod
    def _to_tensor(patch):
        if isinstance(patch, torch.Tensor):
            return patch.float()

        if isinstance(patch, np.ndarray):
            if patch.ndim == 3:
                patch = patch.transpose(2, 0, 1)

            return torch.from_numpy(patch).float()

        raise TypeError(f"Unsupported patch type: {type(patch)}")
    
    def __getitem__(self, idx):
        x_start = int(round(self.coords["x_start"][idx]))
        y_start = int(round(self.coords["y_start"][idx]))
        x_end = int(round(self.coords["x_end"][idx]))
        y_end = int(round(self.coords["y_end"][idx]))

        patch = self._crop_with_padding(
            x_start=x_start,
            y_start=y_start,
            x_end=x_end,
            y_end=y_end,
        )

        exclude = (self.exclude_fn is not None and self.exclude_fn(patch))

        if self.aug_fn is not None:
            patch = self.aug_fn(patch)

        if self.prep_fn is not None:
            patch = self.prep_fn(patch)

        patch = self._to_tensor(patch)

        return {
            "patch": patch,
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_end,
            "y_end": y_end,
            "exclude": exclude,
        }


if __name__ == "__main__":
    wsi = Slide("/mnt/warehouse/Projects/HE/Data/Artifacts Segmentation/TCGA_CompassNMD/svs/TCGA-CV-7099-01A-02-BS2.1e152adb-e0cb-4962-8004-a9d9310c5e30.svs")

    ds = GrandQCDataset(
        slide=wsi,
        read_level=0,
        target_mpp=1,
        patch_size=512,
        pad_value=255,
        prep_fn=None,
        aug_fn=None,
        exclude_fn=ExcludeBackground(bg_value=255, threshold=1.0),
        bbox_list=[BBox([0, 0, 10000, 10000], mag=wsi.level_mag[0], mpp=wsi.level_mpp[0])],
        patch_writer=None,
        overlap=0.75,
        grid_offset=0.5
    )
