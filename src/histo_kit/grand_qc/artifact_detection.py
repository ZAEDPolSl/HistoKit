import os
import numpy as np
import torch
from PIL import Image
from openslide import OpenSlide
from torch.utils.data import DataLoader
from .artifacts import Artifact
from ..grand_qc.dataset import GrandQCDataset
from ..grand_qc.visualisation import make_artifacts_color_map, make_overlay
from ..utils.file_utils import get_basename, save_rescaled
from ..utils.image import gaussian_window
from ..utils.matlab2python import list2cell
from ..utils.wsi import load_wsi_mag, get_regions_location
import scipy.io as sio


def process_single_optimized(slide_file, res_dict_path, batch_size, num_workers,
                             device, model, paths_dict, vis_mag, overlap=0.7, mag_model=10,
                             patch_size=512, mode="gaussian", classes = 8, sigma=None, save_mag = 2.5,
                             save_confidence_maps=True):
    """
    Run optimized GrandQC inference on a whole-slide image (WSI), aggregate patch-wise
    predictions into full-resolution confidence maps, produce visualizations and save
    results to desired output folders.

    This function:
    - loads a WSI and a precomputed tissue/background detection map,
    - extracts patches from tissue bounding boxes using a sliding grid,
    - runs the provided model on batches of patches,
    - merges patch predictions into full-image class confidence maps using a weighting
      (Gaussian or average) window,
    - normalizes the aggregated maps by the sum of weights,
    - produces a hard prediction mask and color visualizations,
    - optionally rescales results to a different magnification and saves both the
      hard masks and per-class confidence maps as MATLAB `.mat` files, and
    - saves an overlay visualization.

    Parameters
    ----------
    slide_file : str or pathlib.Path
        Path to the WSI file (e.g. `.svs`) to process.
    res_dict_path : str or pathlib.Path
        Path to `.mat` file containing precomputed tissue/background detection and
        related metadata used to compute bounding boxes (expected keys used in code:
        ``"mask_bg"``, ``"ind_WSI"``, ``"ratio"``, ``"thr"``).
    batch_size : int
        Batch size for the PyTorch DataLoader used during inference.
    num_workers : int
        Number of worker processes for the DataLoader.
    device : torch.device or str
        Device used for inference (e.g. ``"cuda"`` or ``cpu``).
    model : torch.nn.Module
        Pre-trained segmentation/classification model that accepts a batch of image
        tensors and returns per-pixel class confidence maps (shape ``(N, C, H, W)``).
    paths_dict : dict
        Dictionary with output directory paths. Expected keys (used in function):
        - ``"grandqc_overlay_vis"`` : directory to save overlay PNGs
        - ``"masks_grandqc"`` : directory to save `.mat` with hard masks
        - ``"masks_grandqc_confidence_maps"`` : directory to save per-class `.mat`
    vis_mag : float
        Magnification used for creating visualization overlays (e.g. 2.5, 5, 10).
    overlap : float, optional
        Fractional overlap between sliding patches (default ``0.7``).
    mag_model : float or int, optional
        Working magnification at which the model expects inputs (default ``10``).
    patch_size : int, optional
        Side length of extracted square patches in pixels (default ``512``).
    mode : {'gaussian', 'average'}, optional
        Weighting mode used when aggregating overlapping patch predictions. If
        ``'gaussian'``, a Gaussian window is applied; if ``'average'``, equal weights
        are used (default ``"gaussian"``).
    classes : int, optional
        Number of classes (including background) predicted by the network (default ``8``).
    sigma : float or None, optional
        Standard deviation parameter for the Gaussian window (only relevant when
        ``mode=='gaussian'``). If ``None``, a reasonable default (from gaussian_window)
        is assumed by that helper (default ``None``).
    save_mag : float or int, optional
        Magnification at which final masks/confidence maps are saved (default ``2.5``).
    save_confidence_maps : bool, optional
        If True, save per-class confidence maps as separate `.mat` files in the
        directory pointed by ``paths_dict["masks_grandqc_confidence_maps"]`` (default True).

    Returns
    -------
    dict
        A dictionary summarizing saved results and metadata. Keys include:
        - ``'basename'`` : str, base filename of the processed WSI (without extension)
        - ``'mask_art'`` : list or MATLAB cell (converted inside function) of hard masks
          for tissue regions
        - ``'ind_WSI'`` : value loaded from ``res_dict_path`` (indexes for WSI layers)
        - ``'ratio'`` : value loaded from ``res_dict_path`` (ratio per layer)
        - ``'scale_val'`` : float, final scale factor applied to masks
        - ``'thr'`` : thresholds loaded from ``res_dict_path`` for color channels
        - ``'bbox'`` : bounding boxes used for patch extraction (possibly rescaled)
        (The function also writes multiple `.mat` files and PNGs to disk as a side effect.)

    Notes
    -----
    - The function may produce coordinates outside original image bounds; these are
      handled by cropping and padding logic inside the dataset / merge loop.
    - The caller must ensure that ``paths_dict`` directories exist and are writable.
    - Large WSIs can consume substantial memory; the implementation aggregates patch
      predictions into full-resolution arrays sized (H, W, classes) — ensure sufficient
      RAM is available for the target magnification and image size.

    Examples
    --------
    >>> paths = {
    ...     "grandqc_overlay_vis": "/out/overlays",
    ...     "masks_grandqc": "/out/masks",
    ...     "masks_grandqc_confidence_maps": "/out/conf_maps"
    ... }
    >>> save_info = process_single_optimized(
    ...     "slide.svs",
    ...     "slide_res.mat",
    ...     batch_size=8,
    ...     num_workers=4,
    ...     device="cuda",
    ...     model=my_model,
    ...     paths_dict=paths,
    ...     vis_mag=2.5,
    ...     overlap=0.7,
    ...     mag_model=10,
    ...     patch_size=512
    ... )
    >>> print(save_info["basename"])

    References
    ----------
    This function is based on \ :footcite:p:`Weng2024`

    """

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
    mag_l0 = float(slide.properties["openslide.objective-power"])
    scale_vis = vis_mag / mag_l0
    vis_size = (int(w_l0 * scale_vis), int(h_l0 * scale_vis))

    data = sio.loadmat(res_dict_path)
    tis_det = data["mask_bg"]
    tis_det = np.array(Image.fromarray(tis_det).resize((W, H), Image.Resampling.NEAREST))
    bbox = get_regions_location(tis_det)

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

    del dataset

    pred_mask = np.argmax(raw_mask, axis=2).astype('int8')
    pred_mask = pred_mask[:region.shape[0], :region.shape[1]]

    # remove the rest of bg pixels
    tis_det_bool = tis_det.astype(bool)
    pred_mask[~tis_det_bool] = 0

    # make color visualisation
    artifacts_color_map = Image.fromarray(make_artifacts_color_map(pred_mask))

    # overlay heatmap on the image
    overlay = make_overlay(region, artifacts_color_map, tis_det, vis_size)
    overlay = Image.fromarray(overlay)
    save_rescaled(overlay, vis_size, os.path.join(paths_dict["grandqc_overlay_vis"], f'{basename}.png'))

    del overlay
    del artifacts_color_map
    del weights

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

    list_raw = [
        {Artifact.NORM.name:[]},
        {Artifact.ART_FOLD.name: []},
        {Artifact.ART_DARKSPOT.name: []},
        {Artifact.ART_PEN.name: []},
        {Artifact.ART_EDGE.name: []},
        {Artifact.ART_FOCUS.name: []},
        {Artifact.BG_MODEL.name: []},
    ]

    for n, region_bbox in enumerate(bbox):
        y0, x0, y1, x1 = map(int, region_bbox)

        y0 = max(0, min(y0, h_res))
        y1 = max(0, min(y1, h_res))
        x0 = max(0, min(x0, w_res))
        x1 = max(0, min(x1, w_res))

        pred_mask_region = pred_mask[y0:y1, x0:x1]
        raw_mask_region = raw_mask[y0:y1, x0:x1, :]

        save_dict['mask_art'].append(pred_mask_region)

        if save_confidence_maps:
            for idx, dict_art in enumerate(list_raw):
                # +1, we don't take layer with background thresholding results into account
                dict_art[Artifact(idx+1).name].append(raw_mask_region[:,:,idx+1])

    # convert to cells for matlab
    del pred_mask
    del raw_mask

    save_dict['mask_art'] = list2cell(save_dict['mask_art'])
    sio.savemat(os.path.join(paths_dict["masks_grandqc"], f'{basename}.mat'), save_dict, do_compression=True)

    if save_confidence_maps:
        for idx, dict_art in enumerate(list_raw):
            # save artifacts layer by layer
            dict_art[Artifact(idx+1).name] = list2cell(dict_art[Artifact(idx+1).name])
            sio.savemat(os.path.join(paths_dict["masks_grandqc_confidence_maps"],
                                     f'{basename}_{Artifact(idx).name}.mat'), dict_art, do_compression=True)

    return save_dict