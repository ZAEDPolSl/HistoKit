import os

import cv2
import numpy as np
import torch
from PIL import Image
from openslide import OpenSlide
from torch.utils.data import DataLoader
from ..grand_qc.dataset import GrandQCDataset
from ..grand_qc.visualisation import make_artifacts_color_map, make_overlay
from ..utils.file_utils import get_basename, save_rescaled
from ..utils.image import gaussian_window
from ..utils.matlab2python import list2cell
from ..utils.patches import load_wsi_mag
import scipy.io as sio


def process_single_optimized(slide_file, res_dict_path, batch_size, num_workers,
                             device, model, paths_dict, scale_thumbnail, overlap=0.5, mag_model=10, patch_size=512,
                             mode="gaussian", classes = 8, sigma=None, save_mag = 10):

    # slide basename
    basename = get_basename(slide_file)

    # load slide
    slide = OpenSlide(slide_file)

    # rescale region
    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(slide, mag_model, allow_upscaling=True)
    W, H = region.size
    region = np.array(region)

    # size for visualisations
    w_l0, h_l0 = slide.level_dimensions[0]
    vis_size = (int(w_l0 * scale_thumbnail), int(h_l0 * scale_thumbnail))

    data = sio.loadmat(res_dict_path)
    tis_det = data["mask_bg"]
    tis_det = np.array(Image.fromarray(tis_det).resize((W, H), Image.Resampling.NEAREST))
    bbox = data["bbox"]

    dataset = GrandQCDataset(region, tis_det, bbox,  patch_size, overlap)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    model.eval()

    raw_mask = np.zeros((H, W, classes))
    weights = np.zeros((H, W))
    if mode == "gaussian":
        weight_patch = gaussian_window(patch_size, patch_size, sigma=sigma)
    elif mode == "average":
        weight_patch = np.ones((patch_size, patch_size))

    for batch in dataloader:
        with torch.no_grad():
            images = batch["patch"].to(device)
            pred = model(images).to("cpu").numpy()

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
            gauss_patch_crop = weight_patch[src_y0:src_y0 + h, src_x0:src_x0 + w]  # (h, w)

            assert pred_patch.shape[0] == gauss_patch_crop.shape[0] and pred_patch.shape[1] == gauss_patch_crop.shape[
                1], \
                f"Shape mismatch: pred_patch {pred_patch.shape}, gauss {gauss_patch_crop.shape}"

            raw_mask[dst_y0:dst_y1, dst_x0:dst_x1, :] += pred_patch * gauss_patch_crop[..., None]
            weights[dst_y0:dst_y1, dst_x0:dst_x1] += gauss_patch_crop

    for i in range(classes):
        raw_mask[:, :, i] = np.divide(
            raw_mask[:, :, i],
            weights,
            out=np.zeros_like(raw_mask[:, :, i]),
            where=weights != 0
        )

    pred_mask = np.argmax(raw_mask, axis=2).astype('int8')
    pred_mask = pred_mask[:region.shape[0], :region.shape[1]]

    # remove the rest of bg pixels
    tis_det_bool = tis_det.astype(bool)
    pred_mask[~tis_det_bool] = 0

    # make color visualisation
    artifacts_color_map = Image.fromarray(make_artifacts_color_map(pred_mask))
    save_rescaled(artifacts_color_map, vis_size, os.path.join(paths_dict["grandqc_map_vis"], f'{basename}.png'))

    # overlay heatmap on the image
    overlay = make_overlay(region, artifacts_color_map, tis_det, vis_size)
    overlay = Image.fromarray(overlay)
    save_rescaled(overlay, vis_size, os.path.join(paths_dict["grandqc_overlay_vis"], f'{basename}.png'))

    # save weights as a grayscale image with bboxes and patches visualisation
    coords = dataset.coords
    weights = Image.fromarray(
        ((weights - np.min(weights)) / (np.max(weights) - np.min(weights)) * 255).astype(np.uint8)).convert("RGB")
    weights = np.array(weights)
    for b in bbox:
        y_min, x_min, y_max, x_max = b
        cv2.rectangle(
            weights,
            (x_min, y_min),
            (x_max, y_max),
            color=(255, 0, 255),
            thickness=2
        )

    for x_s, y_s, x_e, y_e in zip(coords["x_start"], coords["y_start"], coords["x_end"], coords["y_end"]):
        cv2.rectangle(
            weights,
            (max(0, x_s), max(0, y_s)),
            (min(weights.shape[1], x_e), min(weights.shape[0], y_e)),
            color=(0, 255, 0),
            thickness=2
        )
    save_rescaled(weights, vis_size, os.path.join(paths_dict["grandqc_vis_weights"], f'{basename}.png'))

    # rescale result to desired magnification
    if mag_model != save_mag:

        scale = float(mag_model) / float(save_mag)

        new_H = int(round(H / scale))
        new_W = int(round(W / scale))
        resized_size = (new_W, new_H)

        pred_mask = np.array(Image.fromarray(pred_mask).resize(resized_size, Image.Resampling.NEAREST))

        raw_mask_rescaled = np.zeros((new_H, new_W, raw_mask.shape[2]), dtype=raw_mask.dtype)
        for c in range(raw_mask.shape[2]):
            img = Image.fromarray(raw_mask[:, :, c])
            img_resized = img.resize(resized_size, Image.Resampling.NEAREST)
            raw_mask_rescaled[:, :, c] = np.array(img_resized)
        raw_mask = raw_mask_rescaled

        bbox = np.array(bbox, dtype=float)
        bbox = np.round(bbox / scale).astype(int)


    mag_l0 = float(slide.properties["openslide.objective-power"])
    h_res, w_res = pred_mask.shape
    scale_val = save_mag/mag_l0

    save_dict = {
        'basename': basename,  # tissue file basename (without .svs extension)
        'mask_art': [],  # mask with artifacts detected by grandQC for given region
        'ind_WSI': data['ind_WSI'],  # indexes for WSI image layers (idx from 1)
        'ratio': data['ratio'],  # ratio for each layer
        'scale_val': scale_val,  # scale factor of masks
        'thr': data['thr'],  # thresholds calculated for R, G, B color channels
        'bbox': bbox # bboxes for tissue regions (indexing from 0)
    }

    save_dict_raw = {
        'raw_mask_art':[], # raw mask with predictions (3D)
    }

    for n, region_bbox in enumerate(bbox):
        y0, x0, y1, x1 = map(int, region_bbox)

        y0 = max(0, min(y0, h_res))
        y1 = max(0, min(y1, h_res))
        x0 = max(0, min(x0, w_res))
        x1 = max(0, min(x1, w_res))

        pred_mask_region = pred_mask[y0:y1, x0:x1]
        raw_mask_region = raw_mask[y0:y1, x0:x1, :]

        save_dict['mask_art'].append(pred_mask_region)
        save_dict_raw['raw_mask_art'].append(raw_mask_region)

        region_color_map = Image.fromarray(make_artifacts_color_map(pred_mask_region))
        new_size = (int(region_color_map.size[0]*scale_thumbnail/scale_val), int(region_color_map.size[1]*scale_thumbnail/scale_val))
        region_color_map = region_color_map.resize(new_size, Image.Resampling.NEAREST)
        region_color_map.save(os.path.join(paths_dict["grandqc_vis_region"], f'{basename}_R{n}.png'))

    # convert to cells for matlab
    save_dict_raw['raw_mask_art'] = list2cell(save_dict_raw['raw_mask_art'])
    save_dict['mask_art'] = list2cell(save_dict['mask_art'])

    sio.savemat(os.path.join(paths_dict["masks_grandqc"], f'{basename}.mat'), save_dict, do_compression=True)
    sio.savemat(os.path.join(paths_dict["masks_grandqc_confidence_maps"], f'{basename}.mat'), save_dict_raw, do_compression=True)

    return save_dict