import os
from PIL import Image
from torch import classes
from .vis import vis_segmentation_results_multiclass
import numpy as np


def dice(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

def jaccard(tp, fp, fn):
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def specificity(tn, fp):
    return tn/(tn+fp) if (tn+fp)>0 else 0.0

def npv(tn, fn):
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

def fdr(tp, fp):
    return fp / (tp + fp) if (tp + fp) > 0 else 0.0

def calc_metrics(tp, tn, fp, fn):
    if tp + tn + fp + fn == 0:
        return {
            "DICE": None,
            "JACCARD": None,
            "PRECISION": None,
            "RECALL": None,
            "ACCURACY": None,
            "F1": None,
            "SPECIFICITY": None,
            "NPV": None,
            "FDR": None,
        }

    precision_val = precision(tp, fp)
    recall_val = recall(tp, fn)
    return {
        "DICE": dice(tp, fp, fn),
        "JACCARD": jaccard(tp, fp, fn),
        "PRECISION": precision_val,
        "RECALL": recall_val,
        "ACCURACY": accuracy(tp, tn, fp, fn),
        "F1": f1_score(precision_val, recall_val),
        "SPECIFICITY": specificity(tn, fp),
        "NPV": npv(tn, fn),
        "FDR": fdr(tp, fp),
    }

def calc_metrics_binary(mask_gt, mask_pred, positive_class=[128, 128, 128]):
    positive_class = np.array(positive_class)
    gt_positive = np.all(mask_gt == positive_class, axis=-1)
    pred_positive = np.all(mask_pred == positive_class, axis=-1)
    tp = np.sum(gt_positive & pred_positive)
    tn = np.sum(~gt_positive & ~pred_positive)
    fp = np.sum(~gt_positive & pred_positive)
    fn = np.sum(gt_positive & ~pred_positive)
    return calc_metrics(tp, tn, fp, fn)


def calc_metrics_multiclass(mask_gt, mask_pred, classes):
    stats = {}

    for class_name, color in classes.items():
        pred_class = np.all(mask_pred == color, axis=-1)
        gt_class = np.all(mask_gt == color, axis=-1)

        tp = np.sum(pred_class & gt_class)
        tn = np.sum(~pred_class & ~gt_class)
        fp = np.sum(pred_class & ~gt_class)
        fn = np.sum(~pred_class & gt_class)

        metrics = calc_metrics(tp, tn, fp, fn)

        stats[class_name] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "Precision": metrics["PRECISION"],
            "Recall": metrics["RECALL"],
            "F1": metrics["F1"],
            "Accuracy": metrics["ACCURACY"],
            "Specificity": metrics["SPECIFICITY"],
            "NPV": metrics["NPV"],
            "FDR": metrics["FDR"],
            "DICE": metrics["DICE"],
            "Jaccard": metrics["JACCARD"],
        }
    return stats


def evaluate_rgb_mask(mask_gt, mask_pred, mask_basename, vis_dir, method="HistoKit (no postprocessing)", 
                      tissue_class=[128, 128, 128], bg_class=[0, 0, 0], multiclass = True, class_dict = None):

    # Visualisation
    img_vis = vis_segmentation_results_multiclass(
        mask_gt=mask_gt,
        mask_pred=mask_pred,
        tissue_class=tissue_class,
        bg_class=bg_class,
    )

    out_path = os.path.join(vis_dir, mask_basename)
    Image.fromarray(img_vis).save(out_path)

    # Binary metrics: tissue = positive class
    #                 bg & art = negative class
    binary_metrics = calc_metrics_binary(
        mask_gt=mask_gt,
        mask_pred=mask_pred,
        positive_class=tissue_class,
    )

    res_dict_binary = {
        "Method": method,
        "Mode": mask_basename,
        "Image": mask_basename,
        **binary_metrics,
    }

    if multiclass and class_dict is not None:
        # Multiclass metrics: one-vs-rest for each RGB class
        res_dict_multiclass = {
            "Method": method,
            "Mode": mask_basename,
            "Image": mask_basename,
        }

        for class_name, class_rgb in class_dict.items():
            class_metrics = calc_metrics_binary(
                mask_gt=mask_gt,
                mask_pred=mask_pred,
                positive_class=class_rgb,
            )

            for metric_name, metric_value in class_metrics.items():
                res_dict_multiclass[f"{class_name}_{metric_name}"] = metric_value

    return res_dict_binary, res_dict_multiclass