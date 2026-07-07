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

def calc_metrics_binary(gt_mas, pred_mask):
    tp = np.sum((gt_mas == 1) & (pred_mask == 1))
    tn = np.sum((gt_mas == 0) & (pred_mask == 0))
    fp = np.sum((gt_mas == 0) & (pred_mask == 1))
    fn = np.sum((gt_mas == 1) & (pred_mask == 0))
    return calc_metrics(tp, tn, fp, fn)

def calc_metrics_multiclass(gt_mask, pred_mask, num_classes):
    metrics_per_class = {}
    for class_id in range(num_classes):
        gt_binary = (gt_mask == class_id).astype(int)
        pred_binary = (pred_mask == class_id).astype(int)
        metrics_per_class[class_id] = calc_metrics_binary(gt_binary, pred_binary)
    return metrics_per_class
