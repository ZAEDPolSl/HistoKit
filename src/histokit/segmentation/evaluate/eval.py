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
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
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
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
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

def calc_metrics_binary(
    mask_gt,
    mask_pred,
    positive_class=[128, 128, 128],
    ignore_mask=None,
):
    positive_class = np.array(positive_class)

    gt_positive = np.all(mask_gt == positive_class, axis=-1)
    pred_positive = np.all(mask_pred == positive_class, axis=-1)

    if ignore_mask is not None:
        gt_positive = gt_positive[~ignore_mask]
        pred_positive = pred_positive[~ignore_mask]

    tp = np.sum(gt_positive & pred_positive)
    tn = np.sum(~gt_positive & ~pred_positive)
    fp = np.sum(~gt_positive & pred_positive)
    fn = np.sum(gt_positive & ~pred_positive)

    metrics = calc_metrics(tp, tn, fp, fn)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        **metrics,
    }


def calc_metrics_multiclass(
    mask_gt,
    mask_pred,
    classes,
    tissue_class=[128, 128, 128],
    bg_class=[0, 0, 0],
    ignore_bg_artifact_confusion=True,
):
    stats = {}

    tissue_class = np.array(tissue_class)
    bg_class = np.array(bg_class)

    gt_tissue = np.all(mask_gt == tissue_class, axis=-1)
    pred_tissue = np.all(mask_pred == tissue_class, axis=-1)

    gt_bg = np.all(mask_gt == bg_class, axis=-1)
    pred_bg = np.all(mask_pred == bg_class, axis=-1)

    gt_artifact = ~gt_tissue & ~gt_bg
    pred_artifact = ~pred_tissue & ~pred_bg

    if ignore_bg_artifact_confusion:
        ignore_mask = (gt_bg & pred_artifact) | (gt_artifact & pred_bg)
    else:
        ignore_mask = None

    for class_name, color in classes.items():
        stats[class_name] = calc_metrics_binary(
            mask_gt=mask_gt,
            mask_pred=mask_pred,
            positive_class=color,
            ignore_mask=ignore_mask,
        )

    return stats


def evaluate_rgb_mask(
    mask_gt,
    mask_pred,
    mask_basename,
    vis_dir,
    method,
    tissue_class=[128, 128, 128],
    bg_class=[0, 0, 0],
    multiclass=True,
    class_dict=None,
):
    img_vis = vis_segmentation_results_multiclass(
        mask_gt=mask_gt,
        mask_pred=mask_pred,
        tissue_class=tissue_class,
        bg_class=bg_class,
    )

    out_path = os.path.join(vis_dir, mask_basename)
    Image.fromarray(img_vis).save(out_path)

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

    res_dict_multiclass = None

    if multiclass:
        if class_dict is None:
            raise ValueError("class_dict cannot be None when multiclass=True")

        multiclass_metrics = calc_metrics_multiclass(
            mask_gt=mask_gt,
            mask_pred=mask_pred,
            classes=class_dict,
            tissue_class=tissue_class,
            bg_class=bg_class,
            ignore_bg_artifact_confusion=True,
        )

        res_dict_multiclass = {
            "Method": method,
            "Mode": mask_basename,
            "Image": mask_basename,
        }

        for class_name, class_metrics in multiclass_metrics.items():
            for metric_name, metric_value in class_metrics.items():
                res_dict_multiclass[f"{class_name}_{metric_name}"] = metric_value

    return res_dict_binary, res_dict_multiclass