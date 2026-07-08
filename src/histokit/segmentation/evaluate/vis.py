import numpy as np

def vis_segmentation_results_binary(
    mask_gt,
    mask_pred,
    positive_class=[128, 128, 128]) -> np.ndarray:

    positive_class = np.array(positive_class)

    gt_positive = np.all(mask_gt == positive_class, axis=-1)
    pred_positive = np.all(mask_pred == positive_class, axis=-1)

    tp = gt_positive & pred_positive
    fp = ~gt_positive & pred_positive
    fn = gt_positive & ~pred_positive
    tn = ~gt_positive & ~pred_positive

    res = np.zeros_like(mask_gt, dtype=np.uint8)

    res[tp] = [0, 255, 0]    # True Positive: Green
    res[fp] = [255, 0, 0]    # False Positive: Red
    res[fn] = [0, 0, 255]    # False Negative: Blue
    res[tn] = [0, 0, 0]      # True Negative: Black
    return res

def vis_segmentation_results_multiclass(mask_gt, mask_pred, tissue_class=[128, 128, 128], bg_class=[0, 0, 0]) -> np.ndarray:
    tissue_class = np.array(tissue_class)
    bg_class = np.array(bg_class)

    gt_tissue = np.all(mask_gt == tissue_class, axis=-1)
    pred_tissue = np.all(mask_pred == tissue_class, axis=-1)

    gt_bg = np.all(mask_gt == bg_class, axis=-1)
    pred_bg = np.all(mask_pred == bg_class, axis=-1)

    same_class = np.all(mask_gt == mask_pred, axis=-1)

    gt_artifact = ~gt_tissue & ~gt_bg
    pred_artifact = ~pred_tissue & ~pred_bg

    tp_tissue = gt_tissue & pred_tissue
    tp_bg = gt_bg & pred_bg
    tp_artifact = same_class & gt_artifact & pred_artifact

    miss_artifact = gt_artifact & pred_artifact & ~same_class

    fp_tissue = ~gt_tissue & pred_tissue

    res = np.zeros_like(mask_gt, dtype=np.uint8)

    res[tp_tissue] = [128, 128, 128]   # correctly assigned tissue
    res[tp_bg] = [0, 0, 0]             # correctly assigned background
    res[tp_artifact] = [0, 255, 0]     # correctly assigned artifact
    res[miss_artifact] = [255, 255, 0] # misclassified artifact
    res[fp_tissue] = [255, 0, 0]       # incorrectly assigned tissue

    return res