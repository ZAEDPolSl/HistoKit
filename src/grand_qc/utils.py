import numpy as np
from PIL import Image
import cv2


def make_overlay(slide, wsi_heatmap_im, vis_size):
    """

    :param slide:
    :param wsi_heatmap_im:
    :param overlay_factor:
    :return:
    """
    slide = Image.fromarray(slide)
    slide_reduced = slide.resize(vis_size, Image.Resampling.NEAREST)
    heatmap_temp = wsi_heatmap_im.resize(slide_reduced.size, Image.Resampling.NEAREST)
    overlay = cv2.addWeighted(np.array(slide_reduced), 0.3, np.array(heatmap_temp), 0.7, 0)
    heatmap_np = np.array(heatmap_temp)
    mask = np.all(heatmap_np == (128, 128, 128), axis=-1)
    overlay[mask] = np.array(slide_reduced)[mask]
    return overlay


def slide_info(slide, model_patch_size, mpp_model, mpp_slide, verbose=False):
    """

    :param verbose:
    :param slide:
    :param model_patch_size:
    :param mpp_model:
    :param mpp_slide:
    :return:
    """
    # Objective power
    try:
        obj_power = slide.properties["openslide.objective-power"]
    except:
        obj_power = 99

    patch_size = int(mpp_model / mpp_slide * model_patch_size)

    # Vendor
    vendor = slide.properties["openslide.vendor"]

    # Extract and save dimensions of level [0]
    dim_l0 = slide.level_dimensions[0]
    width_level_0 = dim_l0[0]
    height_level_0 = dim_l0[1]

    # Calculate number of patches to process
    num_patches_width = int(width_level_0 / patch_size)
    num_patches_height = int(height_level_0 / patch_size)

    # Number of levels
    num_level = slide.level_count

    # Level downsamples
    down_levels = slide.level_downsamples

    # Output BASIC DATA
    if verbose:
        print("")
        print("Basic data about processed whole-slide image")
        print("")
        print("Vendor: ", vendor)
        print("Scan magnification: ", obj_power)
        print("Number of levels: ", num_level)
        print("Level downsamples: ", down_levels)
        print("Microns per pixel (slide) estimated from slide magnification:", mpp_slide)
        print("Height: ", height_level_0)
        print("Width: ", width_level_0)
        print("Model patch size at slide MPP: ", patch_size, "x", patch_size)
        print("Width - number of patches: ", num_patches_width)
        print("Height - number of patches: ", num_patches_height)
        print("Overall number of patches / slide (without tissue detection): ", num_patches_width * num_patches_height)

    return patch_size, num_patches_width, num_patches_height, width_level_0, height_level_0, obj_power
