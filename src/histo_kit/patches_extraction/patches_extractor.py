import os
from torch.utils.data._utils.collate import default_collate
import numpy as np
import torch
from torch.utils.data import Dataset
from src.histo_kit.utils.patches import get_patch_grid
from src.histo_kit.utils.wsi import get_regions_location
from src.histo_kit.grand_qc.artifacts import Artifact
from PIL import Image


def collate_remove_none(batch):
    filtered = [b for b in batch if b is not None]
    if len(filtered) == 0:
        return torch.empty(0, dtype=torch.float32)
    return default_collate(filtered)

class PatchesExtractor(Dataset):


    def __init__(self, regions, wsi_name, out_dir=None,aug=None,out_dir_aug=None, patch_size=512, overlap=0.7, pad_value=Artifact.BG_THR.value, bg_max_percent=0.05, mode = "wsi"):
        self.coords = None
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.regions = regions
        self.overlap = overlap
        self.out_dir = out_dir
        self.mode = mode
        self.bg_max_percent = bg_max_percent
        self.set_coords()
        self.aug = aug
        self.out_dir_aug = out_dir_aug
        self.wsi_name = wsi_name

    def set_coords(self):
        self.coords = {"x_start": [], "y_start": [], "x_end": [], "y_end": [], "region_idx": []}
        for idx, r in enumerate(self.regions):
            mask_bin = np.any(r != self.pad_value, axis=-1).astype(bool)
            bbox_list, masks = get_regions_location(mask_bin)
            region_coords = get_patch_grid(bbox_list, patch_size=self.patch_size, overlap=self.overlap)
            region_coords["region_idx"] = [idx] * len(region_coords["x_start"])
            for key in self.coords:
                self.coords[key].extend(region_coords[key])

    def __len__(self):
        return len(self.coords["x_start"])

    def __getitem__(self, idx):

        region_idx = int(self.coords["region_idx"][idx])
        region = self.regions[region_idx]

        x_start = int(self.coords["x_start"][idx])
        y_start = int(self.coords["y_start"][idx])
        x_end = int(self.coords["x_end"][idx])
        y_end = int(self.coords["y_end"][idx])

        sx0 = max(0, x_start)
        sy0 = max(0, y_start)
        sy1 = min(region.shape[0], y_end)
        sx1 = min(region.shape[1], x_end)

        patch = region[sy0:sy1, sx0:sx1].copy()

        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:

            padded = np.full((self.patch_size, self.patch_size, 3), self.pad_value, dtype=np.uint8)

            paste_x = max(0, -x_start)
            paste_y = max(0, -y_start)

            h_copy = min(patch.shape[0], self.patch_size - paste_y)
            w_copy = min(patch.shape[1], self.patch_size - paste_x)

            if h_copy > 0 and w_copy > 0:
                padded[paste_y:paste_y + h_copy, paste_x:paste_x + w_copy] = patch[:h_copy, :w_copy]
            patch = padded

        if np.all(patch == 255, axis=2).mean()>self.bg_max_percent:
            return None

        if self.out_dir is not None:
            if self.mode == "region":
                path = f"{self.out_dir}/{self.wsi_name}_R_{region_idx}/{x_start}_{y_start}_{x_end}_{y_end}.png"
            elif self.mode == "wsi":
                path = f"{self.out_dir}/{self.wsi_name}/{x_start}_{y_start}_{x_end}_{y_end}.png"

            os.makedirs(os.path.dirname(path), exist_ok=True)
            Image.fromarray(patch).save(path)

        if self.out_dir_aug is not None and self.aug is not None:
            patch_aug = self.aug(Image.fromarray(patch))
            if self.mode == "region":
                path = f"{self.out_dir_aug}/{self.wsi_name}_R_{region_idx}/{x_start}_{y_start}_{x_end}_{y_end}.png"
            elif self.mode == "wsi":
                path = f"{self.out_dir_aug}/{self.wsi_name}/{x_start}_{y_start}_{x_end}_{y_end}.png"

            os.makedirs(os.path.dirname(path), exist_ok=True)
            patch_aug.save(path)

        return

    def extract_patches(self, batch_size=16, num_workers=4, pin_memory=True):
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_remove_none,
            pin_memory=pin_memory
        )

        for _ in dataloader:
            pass

