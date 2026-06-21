import os
import time
import warnings
from .config import GrandQCConfig
from ...collectors.base import OutputKind
import numpy as np
from typing import Dict
import torch
from torch.utils.data import DataLoader
from ....patch_extractors.datasets.grid import GridExtractorDataset
from ....slide.bbox import BBox
from ....slide.mask_utils import scale_mask_to_bbox, merge_regions
from ....slide.slide import Slide
from ...base import Segmenter
from ...utils import get_weights
import segmentation_models_pytorch as smp
from ....savers.base import Saver


class GrandQCSegmenter(Segmenter):

    def __init__(self, 
                 config: GrandQCConfig, 
                 output_collector=None, 
                 saver=None):
        
        self.config = config

        # Visualisations and other partial results
        self.output_collector = (
            output_collector
            if output_collector is not None
            else config.build_output_collector()
        )

        # Final mask saving
        self.saver = (
            saver
            if saver is not None
            else Saver(self.config.saver)
        )

        # Load the GrandQC model
        self.model = torch.load(self.config.model_path,
                                map_location=self.config.device,
                                weights_only=False)
        self.model.to(self.config.device)
        self.model.eval()

        # Precompute the blending weights for patch merging
        self.weight_patch = get_weights(self.config.blending_mode,
                                   self.config.patch_size,
                                   self.config.patch_size,
                                   sigma=self.config.blending_sigma)
    

    def segment(self, 
                slide: Slide, 
                basename: str = "slide", 
                tissue_mask: dict | None = None, 
                verbose: bool = False, 
                save: bool = True) -> Dict:

        if verbose:
            print(f"Segmenting artifacts for slide: {basename} using GrandQC...")

        t0 = time.perf_counter()

        if tissue_mask is None:
            # When there is no tissue mask provided, patches will be 
            # extracted from the entire slide. 

            w, h = slide.get_size_at_mag(self.config.det_mag)

            tissue_mask = {
                "mask": [np.ones((h, w), dtype=bool)],
                "bbox": [np.array([0, 0, w, h])],
                "mag_save": self.config.det_mag,
            }

        result = {
            "basename": basename,
            "method": "GrandQC",
            "type": "artifact_mask",

            "mask": [],
            "bbox": [],
            "raw_mask": [] if self.config.save_raw_mask else None,

            "mag_det": self.config.det_mag,
            "mag_save": self.config.save_mag,

            "mag_l0": slide.mag,
            "mpp_l0": slide.mpp,
            "level_dimensions_0": np.array(slide.level_dimensions[0]),

            "config": self.config.to_hdf5_dict(),
            "time": 0,
        }

        for idx, (mask, bbox) in enumerate(zip(tissue_mask["mask"], tissue_mask["bbox"])):

            if verbose:
                print(f"Processing tissue region {idx + 1}/{len(tissue_mask['mask'])}")

            # 1. Read the tissue region at detection magnification
            region_np = np.array(slide.read_masked_object(
                bbox=bbox,
                mask=mask,
                mag_bbox=tissue_mask["mag_save"],
                mag=self.config.det_mag,
                pad_value=self.config.pad_value,
            ))

            ds = GridExtractorDataset(
                region_np,
                patch_size=self.config.patch_size,
                overlap=self.config.overlap,
                pad_value=self.config.pad_value,
                grid_offset=self.config.grid_offset,
                prep_fn= smp.encoders.get_preprocessing_fn(
                    self.config.encoder,
                    self.config.encoder_weights,
                )
            )

            H, W = region_np.shape[:2]

            # 2. Initialize raw mask and weights

            raw_mask = np.zeros((H, W, self.config.classes),dtype=np.float32)
            weights = np.zeros((H, W), dtype=np.float32)

            loader = DataLoader(
                ds,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

            # 3. Run GrandQC in batches

            for batch in loader:
                with torch.no_grad():
                    images = batch["patch"].to(self.config.device)
                    pred = self.model(images).to("cpu").numpy()

                for i, pred_single in enumerate(pred):
                    pred_hwc = pred_single.transpose(1, 2, 0)

                    orig_x0 = int(batch["x_start"][i])
                    orig_y0 = int(batch["y_start"][i])
                    orig_x1 = int(batch["x_end"][i])
                    orig_y1 = int(batch["y_end"][i])

                    dst_x0 = max(0, orig_x0)
                    dst_y0 = max(0, orig_y0)
                    dst_x1 = min(W, orig_x1)
                    dst_y1 = min(H, orig_y1)

                    h = dst_y1 - dst_y0
                    w = dst_x1 - dst_x0

                    if h <= 0 or w <= 0:
                        continue

                    src_x0 = dst_x0 - orig_x0
                    src_y0 = dst_y0 - orig_y0

                    pred_patch = pred_hwc[
                        src_y0:src_y0 + h,
                        src_x0:src_x0 + w,
                        :,
                    ]

                    weight_patch = self.weight_patch[
                        src_y0:src_y0 + h,
                        src_x0:src_x0 + w,
                    ]

                    raw_mask[
                        dst_y0:dst_y1,
                        dst_x0:dst_x1,
                        :,
                    ] += pred_patch * weight_patch[..., None]

                    weights[
                        dst_y0:dst_y1,
                        dst_x0:dst_x1,
                    ] += weight_patch

            # 4. Normalize the raw mask by the weights to get the final 
            # confidence maps for each class.
            raw_mask = np.divide(
                raw_mask,
                weights[:, :, None],
                out=np.zeros_like(raw_mask),
                where=weights[:, :, None] != 0,
            )

            # 5. Get the predicted mask by taking the argmax across classes
            pred_mask = np.argmax(raw_mask, axis=2).astype(np.int8)

            # 6. Set background pixels (pad_value) to class 0 in the predicted mask
            bg = np.all(region_np == self.config.pad_value, axis=2)

            raw_mask[:, :, 0] = bg.astype(raw_mask.dtype)
            pred_mask[bg] = 0

            # 7. Scale the predicted mask and raw mask to the save magnification.
            bbox_obj = BBox.normalize(bbox, mag=self.config.det_mag)
            bbox_save = bbox_obj.scale(mag=self.config.save_mag)

            if bbox_save.w < 1 or bbox_save.h < 1:
                warnings.warn(
                    f"Skipping region {idx + 1} due to small size after scaling. BBox: {bbox_save}",
                    UserWarning,
                )
                continue

            result["mask"].append(scale_mask_to_bbox(pred_mask, bbox_save))
            result["bbox"].append(bbox_save.numpy())

            if self.config.save_raw_mask:
                result["raw_mask"].append(
                    scale_mask_to_bbox(raw_mask, bbox_save)
                )

            # * Free memory
            del region_np
            del raw_mask
            del pred_mask

        result["time"] = time.perf_counter() - t0

        # 8. Collect outputs for visualization (optional)
        if self.config.collectors:

            masks = []
            bboxes = []
            
            # Visialisation are done for a whole slide, so masks are 
            # scaled and merged to a single mask.

            for mask, bbox in zip(result["mask"], result["bbox"]):
                bbox_vis = BBox.normalize(bbox, mag=self.config.save_mag,
                ).scale(mag=self.config.vis_mag)

                if bbox_save.w < 1 or bbox_save.h < 1:
                    warnings.warn(
                        f"Skipping region {idx + 1} due to small size after scaling. BBox: {bbox_save}",
                        UserWarning,
                    )
                    continue

                masks.append(scale_mask_to_bbox(mask, bbox_vis))
                bboxes.append(bbox_vis.numpy().astype(int))

            w, h = slide.get_size_at_mag(self.config.vis_mag)

            mask_merged = merge_regions(
                masks,
                bboxes,
                shape=(h, w),
            )

            thumb = np.array(slide.read_region(mag=self.config.vis_mag))

            self._collect(
                name="artifact_overlay",
                step="visualisation",
                kind=OutputKind.MASK,
                data=mask_merged,
                basename=basename,
                image=thumb,
                colors=self.config.colors,
            )

        # 9. Save results (optional)
        if save:
            self.saver.save(
                os.path.join(self.config.out_dir, "mask_grandqc"),
                basename,
                result,
            )

        return result
