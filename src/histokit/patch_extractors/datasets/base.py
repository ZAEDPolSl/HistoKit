from abc import abstractmethod
from ...slide.bbox import BBox
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch
from .conditions import ExcludeBackground
from ..patch_writer import PatchImageWriter

class PatchDataset(Dataset):

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
    ):
        self.region = region
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.prep_fn = prep_fn
        self.aug_fn = aug_fn
        self.exclude_fn = exclude_fn
        self.patch_writer = patch_writer

        if bbox_list is None:
            self.bbox_list = [
                BBox([0, 0, region.shape[1], region.shape[0]])
            ]
        else:
            self.bbox_list = [
                bbox if isinstance(bbox, BBox) else BBox(bbox)
                for bbox in bbox_list
            ]

        self.region = region
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.prep_fn = prep_fn
        self.aug_fn = aug_fn
        self.exclude_fn = exclude_fn
        self.patch_writer = patch_writer
        self.coords = {"bbox_idx": [], "x_start": [], "y_start": [], "x_end": [], "y_end": [], "bbox": []}

    @abstractmethod
    def get_coords(self):
        raise NotImplementedError("Subclasses must implement get_coords method")

    def __len__(self):
        return len(self.coords["x_start"])

    def vis(self):
        plt.imshow(self.region)
        for i in range(len(self.coords["x_start"])):
            x_start = int(round(self.coords["x_start"][i]))
            y_start = int(round(self.coords["y_start"][i]))
            x_end = int(round(self.coords["x_end"][i]))
            y_end = int(round(self.coords["y_end"][i]))
            plt.plot([x_start, x_end, x_end, x_start, x_start],
                     [y_start, y_start, y_end, y_end, y_start], 'r-')
        plt.show()

    def _crop_with_padding(self, x_start, y_start, x_end, y_end):
        h, w = self.region.shape[:2]

        sx0 = max(0, x_start)
        sy0 = max(0, y_start)
        sx1 = min(w, x_end)
        sy1 = min(h, y_end)

        patch = self.region[sy0:sy1, sx0:sx1]

        if patch.shape[:2] == (self.patch_size, self.patch_size):
            return patch

        padded = np.full(
            (self.patch_size, self.patch_size, self.region.shape[2]),
            self.pad_value,
            dtype=self.region.dtype,
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

        return padded

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

        if self.patch_writer is not None:
            self.patch_writer(
                patch=patch,
                x_start=x_start,
                y_start=y_start,
                x_end=x_end,
                y_end=y_end,
                exclude=exclude
            )

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