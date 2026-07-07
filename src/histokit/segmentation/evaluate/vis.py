import numpy as np

def vis_segmentation_results_binary(mask_gt, mask_pred) -> np.ndarray:
    tp = mask_gt & mask_pred
    fp = ~mask_gt & mask_pred
    fn = mask_gt & ~mask_pred
    tn = ~mask_gt & ~mask_pred
    res = np.zeros((*mask_gt.shape, 3), dtype=np.uint8)

    res[tp] = [0, 255, 0]  # True Positive: Green
    res[fp] = [255, 0, 0]  # False Positive: Red
    res[fn] = [0, 0, 255]  # False Negative: Blue   
    res[tn] = [0, 0, 0]  # True Negative: Black

    return res

def vis_segmentation_results_multiclass(mask_gt, mask_pred, tissue_class = 1, bg_class = 0) -> np.ndarray:

    # correctly assingned tissue - gray
    tp_tissue = (mask_gt == tissue_class) & (mask_pred == tissue_class)

    # correctly assingned bg - black
    tp_bg = (mask_gt == bg_class) & (mask_pred == bg_class)

    # correctly assingned artifact - green
    tp_artifact = (mask_gt == mask_pred) & (mask_gt != tissue_class) & (mask_gt != bg_class)

    # missclassified artifact - yellow
    miss_artifact = (mask_gt != mask_pred) & (mask_gt != tissue_class)

    # incorrectly assigned tissue (where was bg or artifact but predicted as tissue) - red
    fp_tissue = (mask_gt != tissue_class) & (mask_pred == tissue_class)

    # Create an RGB image to visualize the results
    res = np.zeros((*mask_gt.shape, 3), dtype=np.uint8)

    res[tp_tissue] = [128, 128, 128]  
    res[tp_bg] = [0, 0, 0]  
    res[tp_artifact] = [0, 255, 0]  
    res[miss_artifact] = [255, 255, 0]  
    res[fp_tissue] = [255, 0, 0] 

    return res
