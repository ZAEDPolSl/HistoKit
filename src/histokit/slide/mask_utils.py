import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from ..slide.bbox import BBox

def split_regions(
    mask: np.ndarray,
    min_area: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got shape {mask.shape}"
        )

    labeled = label(mask != 0, connectivity=2)

    masks = []
    bboxes = []

    for region in regionprops(labeled):
        if min_area is not None and region.area < min_area:
            continue

        min_row, min_col, max_row, max_col = region.bbox

        region_labeled = labeled[min_row:max_row, min_col:max_col]
        region_mask = mask[min_row:max_row, min_col:max_col].copy()

        region_mask[region_labeled != region.label] = 0

        bbox = np.array(
            [
                min_col,
                min_row,
                max_col - min_col,
                max_row - min_row,
            ],
            dtype=int,
        )

        masks.append(region_mask)
        bboxes.append(bbox)

    if len(bboxes) == 0:
        return [], np.empty((0, 4), dtype=int)

    return masks, np.stack(bboxes).astype(int)

def merge_regions(masks, bboxes, shape):
    if len(masks) == 0:
        return np.zeros(shape, dtype=np.uint8)

    dtype = masks[0].dtype

    if masks[0].ndim == 2:
        out_shape = shape
    elif masks[0].ndim == 3:
        out_shape = (*shape, masks[0].shape[2])
    else:
        raise ValueError(f"Unsupported mask shape: {masks[0].shape}")

    merged = np.zeros(out_shape, dtype=dtype)

    for region_mask, bbox in zip(masks, bboxes):
        x, y, w, h = bbox.astype(int)

        x0, y0 = x, y
        x1, y1 = x + w, y + h

        roi = merged[y0:y1, x0:x1]

        hh = min(roi.shape[0], region_mask.shape[0])
        ww = min(roi.shape[1], region_mask.shape[1])

        roi = roi[:hh, :ww]
        region_crop = region_mask[:hh, :ww]

        if region_crop.ndim == 2:
            idx = region_crop != 0
            roi[idx] = region_crop[idx]

        elif region_crop.ndim == 3:
            idx = np.any(region_crop != 0, axis=2)
            roi[idx, :] = region_crop[idx, :]

    return merged

def scale_mask_to_bbox(mask: np.ndarray, bbox: BBox) -> np.ndarray:

    new_size = (int(bbox.w), int(bbox.h))

    if new_size[0] < 1 or new_size[1] < 1:
        raise ValueError(f"Invalid bbox size: {new_size}")

    if mask.ndim == 2:
        return np.array(
            Image.fromarray(mask).resize(new_size, Image.Resampling.NEAREST)
        )

    if mask.ndim == 3:
        channels = [
            np.array(
                Image.fromarray(mask[:, :, c]).resize(
                    new_size,
                    Image.Resampling.NEAREST,
                )
            )
            for c in range(mask.shape[2])
        ]
        return np.stack(channels, axis=2).astype(mask.dtype)

    raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")

def rescale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    
    new_size = (int(mask.shape[1] * scale), int(mask.shape[0] * scale))
    
    if new_size[0] < 1 or new_size[1] < 1:
        raise ValueError(f"Invalid bbox size: {new_size}")
    
    if mask.ndim == 2:
        return np.array(
            Image.fromarray(mask).resize(new_size, Image.Resampling.NEAREST)
        )

    if mask.ndim == 3:
        channels = [
            np.array(
                Image.fromarray(mask[:, :, c]).resize(
                    new_size,
                    Image.Resampling.NEAREST,
                )
            )
            for c in range(mask.shape[2])
        ]
        return np.stack(channels, axis=2).astype(mask.dtype)

    raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")