import numpy as np
from PIL import Image
from src.histo_kit.utils.wsi import get_regions_location

def convert_mask_grandqc(mask_qc, region, des_mag, art_include=None, mode="region"):

    if art_include is None:
        art_include = []

    H, W = region.shape[:2]
    new_mask_qc = np.zeros((H, W), dtype=np.uint8)

    # scaling factor
    scale_val = mask_qc["scale_val"][0]
    scale_val_reg = des_mag / mask_qc["mag_l0"][0]
    scale_factor = scale_val_reg / scale_val

    new_bboxes = []
    # check if bbox info is available
    if "bbox" in mask_qc and mask_qc["bbox"] is not None:
        for bbox in mask_qc["bbox"]:
            y0, x0, y1, x1 = bbox

            # scale
            y0 = int(y0 * scale_factor)
            x0 = int(x0 * scale_factor)
            y1 = int(y1 * scale_factor)
            x1 = int(x1 * scale_factor)

            # clip into region
            y0 = max(0, y0)
            x0 = max(0, x0)
            y1 = min(H, y1)
            x1 = min(W, x1)

            if y1 > y0 and x1 > x0:  # ensure valid box
                new_bboxes.append([y0, x0, y1, x1])
            else:
                new_bboxes.append(None)
    else:
        Exception("Bounding box information is missing in the mask_qc.")

    tissue_regions = []

    for idx, region_mask in enumerate(mask_qc["mask_art"][0]):
        bbox = new_bboxes[idx]

        if bbox is None:
            continue

        y0, x0, y1, x1 = bbox

        region_h = y1 - y0
        region_w = x1 - x0

        # resize mask to target region size
        resized_mask = np.array(
            Image.fromarray(region_mask).resize(
                (int(region_w), int(region_h)),
                resample=Image.NEAREST
            )
        )

        # remove unwanted artifact classes
        if len(art_include) > 0:
            resized_mask[~np.isin(resized_mask, art_include)] = 0

        # skip empty mask
        if resized_mask.sum() == 0:
            continue

        if mode == "region":
            # apply mask only inside bbox
            mask_bool = resized_mask > 0
            region_crop = region[y0:y1, x0:x1, :].copy()

            # zero out background
            region_crop[~mask_bool] = 255

            tissue_regions.append(region_crop)

        elif mode == "wsi":
            # fill in global mask
            new_mask_qc[y0:y1, x0:x1][resized_mask > 0] = resized_mask[resized_mask > 0]

        else:
            raise ValueError("Mode must be 'wsi' or 'region'.")


    if mode == "wsi":
        region_out = region.copy()
        region_out[new_mask_qc == 0] = 255
        tissue_regions = [region_out]

    return tissue_regions, new_bboxes

def convert_mask_bg(mask_bg, region, des_mag, art_include=None,mode="region"):
    H, W = region.shape[:2]

    mask_pil = Image.fromarray(mask_bg["mask_bg"])
    mask_res = np.array(mask_pil.resize((int(W), int(H)), resample=Image.NEAREST), dtype=np.uint8)

    bbox, masks_regions = get_regions_location(mask_res)
    tissue_regions = []

    if mode == "region":
        for bb, mask in zip(bbox, masks_regions):
            y0, x0, y1, x1 = bb

            # remove unwanted artifact classes
            if len(art_include) > 0:
                mask[~np.isin(mask, art_include)] = 0

            region_crop = region[y0:y1, x0:x1, :].copy()

            # zero out background
            region_crop[~mask] = 255
            tissue_regions.append(region_crop)
    elif mode == "wsi":
        region_out = region.copy()
        region_out[mask_res == 0] = 255
        tissue_regions = [region_out]
    else:
        raise ValueError("Mode must be 'wsi' or 'region'.")

    return tissue_regions, bbox








