import os
from PIL import Image
from math import ceil, floor
import numpy as np
from openslide import OpenSlide
from tqdm import tqdm


def patch_wsi(region, patch_size, save_folder, bg_percent, overlap=0, extract_type="valid"):
    assert 0 <= overlap < 100, "overlap must be in range [0,100)."
    assert 0.0 <= bg_percent <= 1.0, "bg_percent must be in range [0,1]."
    assert region.ndim == 3 and region.shape[2] == 3, "region_rgb must be a RGB image"

    stride = max(int(round(patch_size * (1.0 - overlap / 100.0))), 1)

    H, W = region.shape[:2]

    if extract_type == "valid":
        pad_mode = None

    if pad_mode is None:
        num_x = floor((W - patch_size) / stride) + 1 if W >= patch_size else 0
        num_y = floor((H - patch_size) / stride) + 1 if H >= patch_size else 0
    else:
        num_x = ceil((W - patch_size) / stride) + 1 if W > patch_size else 1
        num_y = ceil((H - patch_size) / stride) + 1 if H > patch_size else 1

    if pad_mode is not None:
        needed_w = (num_x - 1) * stride + patch_size
        needed_h = (num_y - 1) * stride + patch_size
        pad_right = max(0, needed_w - W)
        pad_bottom = max(0, needed_h - H)
        pad_left = 0
        pad_top = 0

        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        if pad_mode == "constant":
            region = np.pad(region, pad_spec, mode=pad_mode, constant_values=255)
        else:
            region = np.pad(region, pad_spec, mode=pad_mode)

    os.makedirs(save_folder, exist_ok=True)

    with tqdm(total=num_x * num_y) as pbar:
        for ix in range(num_x):
            x = ix * stride
            for iy in range(num_y):
                y = iy * stride
                patch = region[y:y + patch_size, x:x + patch_size].copy()

                if np.sum(np.all(patch == 255, axis=2)) / (patch_size * patch_size) < bg_percent:
                    Image.fromarray(patch).save(os.path.join(save_folder, f"patch_{ix}_{iy}.png"))
                pbar.update(1)



def load_wsi_mag(wsi, desired_mag, rescale_method = Image.LANCZOS, verbose = False, allow_upscaling = True):
    """"
    Rescale the WSI.
    :param wsi: OpenSlide wsi object
    :param desired_mag: Desired slide magnification.
    :param rescale_method: Method to downscale WSI if the desired magnification is not available. You can choose from 
    PIL.Image.Resampling (Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.LANCZOS, Image.NEAREST)
    :param allow_upscaling: Allow for image upscaling when desired magnification is higher than the highest magnification available.
    :param verbose: print info messages or not
    :return: rescaled wsi region, scale_val - scale value, when the magnification of level with the highest magnification
    is 40x and we want to take image at magnification 10x scale_val is equal to 40x/10x=4, info - information about rescaling, mpp_slide - approximated slide mpp
    """
    
    ratio = wsi.level_downsamples
    mag_l0 = float(wsi.properties["openslide.objective-power"])
    mag_layers = [round(mag_l0/r, 2) for r in ratio]
    mpp_slide = 10 / mag_layers[0] # approximated slide mpp
    
    if desired_mag in mag_layers:
        info = "Desired magnification is available"
        mag_idx = mag_layers.index(desired_mag)
        w, h = wsi.level_dimensions[mag_idx]
        region = wsi.read_region((0, 0), mag_idx, (w, h))
        scale_val = ratio[mag_idx]
    else:
        if desired_mag > mag_l0:
            info = "Desired slide magnification is larger than available, image will be magnified from the highest magnification available."
            if not allow_upscaling:
                raise ValueError("The desired magnification is smaller than the highest magnification available. "
                                 "The parameter allow_upscaling is set to False, so the image will not be upscaled. "
                                 "If you want to upscale the image, set the parameter allow_upscaling to True. ")
        else:
            info = "Desired resolution is not available, image will be rescaled from the highest magnification available."
        mag_idx = 0  # get the highest magnification and rescale
        w0, h0 = wsi.level_dimensions[mag_idx]
        region = wsi.read_region((0, 0), mag_idx, (w0, h0))
        scale_val = desired_mag/ mag_l0
        region = region.resize((int(w0 * scale_val), int(h0 * scale_val)), rescale_method)

    # convert RGBA to RGB
    region = region.convert("RGB")

    if verbose:
        print(info)
        
    return region, scale_val, info, mpp_slide, ratio

def read_region(wsi, art_mask, bbox, scale_val,ratio, tiss_stats, idx_type = "python"):
    """
    Apply artifacts mask to WSI for a given region. Loads only regions defined by art_mask parameter at coordinates defined by bounding box.
    :param wsi: wsi image
    :param art_mask: artifacts mask from grand QC
    :param bbox: bounding box of region
    :param scale_val: scaling factor
    :param idx_type: type of indexing for artifacts mask (matlab for indexing from 1 and python for indexing from 0)
    :return: region with applied mask
    """
    pass










    
    

    





