import os


def check_gt_pred_exists(gt_path, pred_path):
    return True if os.path.exists(gt_path) and os.path.exists(pred_path) else False

def check_gt_pred_folders(gt_folder, pred_folder, use_ext = False):
    
    gt_files = os.listdir(gt_folder)
    pred_files = os.listdir(pred_folder)

    gt_files = [os.path.splitext(f)[0] for f in gt_files]
    pred_files = [os.path.splitext(f)[0] for f in pred_files]

    if not use_ext:
        gt_files = [os.path.splitext(f)[0] for f in gt_files]
        pred_files = [os.path.splitext(f)[0] for f in pred_files]

    return set(gt_files) == set(pred_files)

def check_gt_pred_img(gt_folder, pred_folder, use_ext = False):
    
    gt_files = os.listdir(gt_folder)
    pred_files = os.listdir(pred_folder)

    gt_files = [os.path.splitext(f)[0] for f in gt_files]
    pred_files = [os.path.splitext(f)[0] for f in pred_files]

    if not use_ext:
        gt_files = [os.path.splitext(f)[0] for f in gt_files]
        pred_files = [os.path.splitext(f)[0] for f in pred_files]

    return set(gt_files) == set(pred_files)