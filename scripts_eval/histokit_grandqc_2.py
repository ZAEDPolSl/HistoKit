from skimage.morphology import remove_small_holes
import skimage
import os
import numpy as np
from PIL import Image
from histokit.savers import HDF5Saver
from tqdm import tqdm
from skimage.morphology import binary_opening, binary_closing, disk
from scipy.ndimage import binary_fill_holes

classes = {
    "Tissue": [128, 128, 128],
    "Background": [0, 0, 0],
    "Fold": [255, 99, 71],
    "Dark.Spot": [0, 255, 0],
    "Pen": [255, 0, 0],
    "Edge": [255, 0, 255],
    "Out.Of.Focus": [75, 0, 130],
}

organs = ["Colon"]
target_mag = 10
for o in organs:
    gt_dir = f"/mnt/warehouse/Projects/HE/Data/Artifacts Segmentation/GrandQC Test Dataset/PreprocessedDataset/{o}/10x/gt_mask"
    main_dir = f"/mnt/warehouse/Projects/HE/Data/Artifacts Segmentation/GrandQC Test Dataset/PreprocessedDataset/{o}/10x/Results/Histokit_30_06_2026/grid_search"
    dirs_img = os.listdir(main_dir)
    folders_processed = [os.path.join(main_dir,f) for f in dirs_img]

    for folder in  tqdm(folders_processed, desc="Processing folders"):

        print("Processing folder:", folder)
        mask_dir = os.path.join(folder, "artifact_detection/grandqc/masks")
        saver = HDF5Saver()
        parsed_dir = os.path.join(folder, "artifact_detection/grandqc/masks_cropped_numeric")
        parsed_color_dir = os.path.join(folder, "artifact_detection/grandqc/masks_cropped_color")
        postprocessed_dir = os.path.join(folder, "artifact_detection/grandqc/masks_cropped_color_postprocessed")
        os.makedirs(parsed_color_dir, exist_ok=True)
        os.makedirs(parsed_dir, exist_ok=True)
        os.makedirs(postprocessed_dir, exist_ok=True)

        for m in os.listdir(mask_dir):
            try:
                name = m.split(".h5")[0]
                dict_mask = saver.load(os.path.join(mask_dir, m))
                size = dict_mask['level_dimensions_0']
                mag = dict_mask['mag_l0']
                factor = target_mag / mag
                size = np.round(size * factor).astype(int)
                slide = np.zeros((size[1], size[0]), dtype=np.uint8)

                for b, mask in zip(dict_mask["bbox"], dict_mask["mask"]):
                    x, y, w, h = [float(v) for v in b]

                    x0 = int(np.floor(x))
                    y0 = int(np.floor(y))
                    x1 = int(np.ceil(x + w))
                    y1 = int(np.ceil(y + h))

                    mask = mask.astype(np.uint8)

                    x0_clip = max(0, x0)
                    y0_clip = max(0, y0)
                    x1_clip = min(x1, slide.shape[1])
                    y1_clip = min(y1, slide.shape[0])

                    if x1_clip <= x0_clip or y1_clip <= y0_clip:
                        continue

                    roi = slide[y0_clip:y1_clip, x0_clip:x1_clip]

                    mask_x0 = x0_clip - x0
                    mask_y0 = y0_clip - y0

                    roi_h, roi_w = roi.shape[:2]

                    mask_crop = mask[
                        mask_y0:mask_y0 + roi_h,
                        mask_x0:mask_x0 + roi_w
                    ]

                    common_h = min(roi.shape[0], mask_crop.shape[0])
                    common_w = min(roi.shape[1], mask_crop.shape[1])

                    roi = roi[:common_h, :common_w]
                    mask_crop = mask_crop[:common_h, :common_w]

                    con = (roi == 0) & (mask_crop != 0)
                    roi[con] = mask_crop[con]

                    slide[
                        y0_clip:y0_clip + common_h,
                        x0_clip:x0_clip + common_w
                    ] = roi

                gt_mask = Image.open(os.path.join(gt_dir, f"{name}.png"))
                slide_img = Image.fromarray(slide)

                if slide_img.size != gt_mask.size:
                    slide_img = slide_img.resize(gt_mask.size, Image.NEAREST)

                slide_img.save(os.path.join(parsed_dir, f"{name}.png"))

                slide = np.array(slide_img)
                gt_np = np.array(gt_mask)
                gt_bg_mask = np.all(gt_np == (0, 0, 0), axis=-1)

                mask_pred = np.zeros((slide.shape[0], slide.shape[1], 3), dtype=np.uint8)
                mask_pred[slide == 0] = classes["Tissue"] # no bg class for grandqc test dataset
                mask_pred[slide == 1] = classes["Tissue"]
                mask_pred[slide == 2] = classes["Fold"]
                mask_pred[slide == 3] = classes["Dark.Spot"]
                mask_pred[slide == 4] = classes["Pen"]
                mask_pred[slide == 5] = classes["Edge"]
                mask_pred[slide == 6] = classes["Out.Of.Focus"]
                mask_pred[slide == 7] = classes["Tissue"] # no bg class for grandqc test dataset
                mask_pred[gt_bg_mask] = classes["Background"] # add gt background class to the prediction (we won't count that)

                Image.fromarray(mask_pred).save(os.path.join(parsed_color_dir, f"{name}.png"))

                # HistoKit postprocessing
                selem = disk(3)
                edge = np.all(mask_pred == classes["Edge"], axis=-1)
                edge = skimage.morphology.opening(edge, footprint=selem)
                edge = skimage.morphology.closing(edge, footprint=selem)
                edge = binary_fill_holes(edge)
                mask_pred[edge] = classes["Edge"]

                tissue = np.all(mask_pred == classes["Tissue"], axis=-1)
                tissue_filled = remove_small_holes(tissue, max_size=int(0.001 * tissue.shape[0] * tissue.shape[1]))

                holes = tissue_filled & ~tissue
                mask_pred[holes] = classes["Tissue"]

                oof = np.all(mask_pred == classes["Out.Of.Focus"], axis=-1)
                bg = np.all(mask_pred == classes["Background"], axis=-1)

                oof_processed = skimage.morphology.opening(oof, footprint=selem)
                oof_processed = skimage.morphology.closing(oof_processed, footprint=selem)
                oof_processed = oof_processed & ~bg
                oof_processed = remove_small_holes(oof_processed, max_size=int(0.001 * tissue.shape[0] * tissue.shape[1]))

                mask_pred[oof_processed] = classes["Out.Of.Focus"]
                mask_pred[gt_bg_mask] = classes["Background"] # add gt background class to the prediction (we won't count that)

                Image.fromarray(mask_pred).save(os.path.join(postprocessed_dir, f"{name}.png"))

            except Exception as e:
                print(f"Error processing {m}")
                print(e)