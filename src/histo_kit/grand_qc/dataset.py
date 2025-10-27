from math import ceil
import numpy as np
from torch.utils.data import Dataset
from ..grand_qc.artifacts import Artifact
import segmentation_models_pytorch as smp
from ..utils.image import to_tensor_x

class GrandQCDataset(Dataset):

    def __init__(self, region, bg, bbox_list, patch_size=512, overlap=0.2, pad_value=Artifact.BG_THR.value, encoder='timm-efficientnet-b0', weights="imagenet"):

        self.bg = bg
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.region = region
        self.bg = bg
        self.prep_fn = smp.encoders.get_preprocessing_fn(encoder, weights)
        self.coords = get_patch_grid(bbox_list, patch_size=patch_size, overlap=overlap)

    def __len__(self):
        return len(self.coords["x_start"])

    def preprocess(self, img):
        x = self.prep_fn(img)
        x = to_tensor_x(x)
        return x

    def __getitem__(self, idx):

        x_start = int(self.coords["x_start"][idx])
        y_start = int(self.coords["y_start"][idx])
        x_end = int(self.coords["x_end"][idx])
        y_end = int(self.coords["y_end"][idx])

        sx0 = max(0, x_start)
        sy0 = max(0, y_start)
        sy1 = min(self.region.shape[0], y_end)
        sx1 = min(self.region.shape[1], x_end)

        patch = self.region[sy0:sy1, sx0:sx1]
        bg_patch = self.bg[sy0:sy1, sx0:sx1]

        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:

            padded = np.full((self.patch_size, self.patch_size, 3), self.pad_value, dtype=np.uint8)
            padded_bg = np.full((self.patch_size, self.patch_size), self.pad_value, dtype=np.uint8)

            paste_x = max(0, -x_start)
            paste_y = max(0, -y_start)

            h_copy = min(patch.shape[0], self.patch_size - paste_y)
            w_copy = min(patch.shape[1], self.patch_size - paste_x)

            if h_copy > 0 and w_copy > 0:
                padded[paste_y:paste_y + h_copy, paste_x:paste_x + w_copy] = patch[:h_copy, :w_copy]
                padded_bg[paste_y:paste_y + h_copy, paste_x:paste_x + w_copy] = bg_patch[:h_copy, :w_copy]
            patch = padded
            bg_patch = padded_bg

        res_dict = {
            "patch": self.preprocess(patch),
            "patch_bg": bg_patch,
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_end,
            "y_end": y_end,
            "all_bg": np.all(bg_patch == self.pad_value)
        }

        return res_dict

def get_patch_grid(regions, patch_size=256, overlap=0.9):

    coords = {"x_start": [], "y_start": [], "x_end": [], "y_end": []}

    for bbox in regions:

        stride = max(int(round(patch_size * (1.0 - overlap))), 1)

        y_0 = bbox[0] - stride
        x_0 = bbox[1] - stride

        tis_h = bbox[2] - y_0  # y_max - y_min
        tis_w = bbox[3] - x_0  # x_max - x_min

        num_x = ceil((tis_w - patch_size) / stride) + 1 if tis_w > patch_size else 1
        num_y = ceil((tis_h - patch_size) / stride) + 1 if tis_h > patch_size else 1

        for ix in range(num_x):
            x = x_0 + ix * stride
            for iy in range(num_y):
                y = y_0 + iy * stride

                coords["x_start"].append(x)
                coords["y_start"].append(y)
                coords["x_end"].append(x + patch_size)
                coords["y_end"].append(y + patch_size)

    return coords




