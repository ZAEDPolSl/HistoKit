import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from ....patch_extractors.datasets.grid import GridExtractorDataset
from ....slide.bbox import BBox
from ....slide.mask_utils import rescale_mask, merge_regions
from ....slide.slide import Slide
from ...tissue.gamred.config import GrandQCConfig
from ...tissue.base import PatchSegmenter
from ...utils import get_weights
import segmentation_models_pytorch as smp
from ....savers.base import Saver


class GrandQCSegmenter(PatchSegmenter):

    def __init__(self, config: GrandQCConfig):
        self.config = config

        self.model = torch.load(self.config.model_path,
                                map_location=self.config.device)
        self.model.to(self.config.device)
        self.model.eval()

        self.preprocess = smp.encoders.get_preprocessing_fn(
            self.config.encoder,
            self.config.encoder_weights,
        )
        self.saver = Saver(self.config.saver)

        self.weight_patch = get_weights(self.config.blending_mode,
                                   self.config.patch_size,
                                   self.config.patch_size,
                                   sigma=self.config.blending_sigma)


    def prep_fn(self, img):
        return self.preprocess(img)

    def run_pipeline_single(self, path:str, path_mask:str = None) -> dict:
        t0 = time.perf_counter()
        slide = Slide(path)
        basename = os.path.basename(path)

        if path_mask is not None:
            mask_data = self.saver.load(path_mask)
        else:
            w, h = slide.get_size_at_mag(self.config.det_mag)
            mask_data = {
                "mask": [np.ones(slide.level_dimensions[-1], dtype=bool)],
                "bbox": [np.array([0, 0, w, h])],
                "mag_save": self.config.det_mag
            }

        result ={
            "basename": os.path.basename(path),
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
            "time": 0
        }

        for idx, (mask, bbox) in enumerate(zip(mask_data["mask"], mask_data["bbox"])):

            region = slide.read_masked_object(bbox=bbox,
                                              mask=mask,
                                              mag_bbox=mask_data["mag_save"],
                                              mag=self.config.det_mag,
                                              pad_value=self.config.pad_value)



            region_np = np.array(region)

            ds = GridExtractorDataset(
                region_np,
                patch_size=self.config.patch_size,
                overlap=self.config.overlap,
                pad_value=self.config.pad_value,
                grid_offset=self.config.grid_offset,
                prep_fn=self.prep_fn,
            )

            H, W = region_np.shape[0], region_np.shape[1]
            raw_mask = np.zeros((H, W, self.config.classes))

            loader = DataLoader(ds, batch_size=self.config.batch_size,
                                num_workers=self.config.num_workers,
                                pin_memory=True)

            weights = np.zeros((H, W))

            for batch in loader:
                with torch.no_grad():
                    images = batch["patch"].to(self.config.device)
                    pred = self.model(images).to("cpu").numpy()

                for i, pred in enumerate(pred):

                    pred_hwc = pred.transpose(1, 2, 0)

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

                    pred_patch = pred_hwc[src_y0:src_y0 + h, src_x0:src_x0 + w, :]  # (h, w, C)
                    gauss_patch_crop = self.weight_patch[src_y0:src_y0 + h, src_x0:src_x0 + w]  # (h, w)

                    raw_mask[dst_y0:dst_y1, dst_x0:dst_x1, :] += pred_patch * gauss_patch_crop[..., None]
                    weights[dst_y0:dst_y1, dst_x0:dst_x1] += gauss_patch_crop

            raw_mask = np.divide(
                raw_mask,
                weights[:, :, None],
                out=np.zeros_like(raw_mask),
                where=weights[:, :, None] != 0
            )

            pred_mask = np.argmax(raw_mask, axis=2).astype('int8')
            pred_mask = pred_mask[:H, :W]

            bg = np.all(region_np == 0, axis=2)
            raw_mask[:, :, 0] = bg

            # remove the rest of bg pixels
            pred_mask[(bg.astype(bool))] = 0

            bbox = BBox.normalize(bbox, mag=self.config.det_mag)
            bbox = bbox.scale(mag=self.config.save_mag)
            result["mask"].append(rescale_mask(pred_mask, bbox))
            result["bbox"].append(bbox.numpy())

            if self.config.save_raw_mask:
                result["raw_mask"].append(rescale_mask(raw_mask, bbox))

            del region_np
            del region


        result["elapsed_time"] = time.perf_counter() - t0

        if len(self.config.visualisation_steps):

            masks = []
            bboxes = []

            for m, b in zip(result["mask"], result["bbox"]):
                b = BBox.normalize(b, mag=self.config.save_mag).scale(mag=self.config.vis_mag)
                masks.append(rescale_mask(m, b))
                bboxes.append(b.numpy().astype(int))

            w, h = slide.get_size_at_mag(self.config.vis_mag)
            mask_merged = merge_regions(masks, bboxes, shape=(h, w))

            vis_dict = {
                "mask": mask_merged,
                "colors": self.config.colors,
                "basename": basename
            }

            for step in self.config.visualisation_steps:
                step(data=vis_dict,
                     slide=slide,
                     save_dir=self.config.out_dir)


        self.saver.save(self.config.out_dir, basename, result)

        return result
