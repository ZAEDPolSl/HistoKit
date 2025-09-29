import numpy as np
from PIL import Image
import cv2
from math import ceil

def make_overlay(slide, wsi_heatmap_im, bg_mask, vis_size):
    """
    Function to overlay GrandQC results on the small WSI thumbnail
    :param vis_size: size of a thumbnail
    :param slide: WSI slide (OpenSlide)
    :param wsi_heatmap_im: mask with artifacts detection results
    :return: overlay - overlay image
    """
    slide = Image.fromarray(slide)
    slide_reduced = slide.resize(vis_size, Image.Resampling.NEAREST)
    heatmap_temp = wsi_heatmap_im.resize(slide_reduced.size, Image.Resampling.NEAREST)
    overlay = cv2.addWeighted(np.array(slide_reduced), 0.3, np.array(heatmap_temp), 0.7, 0)
    contours, _ = cv2.findContours(bg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heatmap_np = np.array(heatmap_temp)
    mask = np.all(heatmap_np == (128, 128, 128), axis=-1)
    overlay[mask] = np.array(slide_reduced)[mask]
    overlay = cv2.drawContours(overlay, contours, -1, (0, 0, 255), 5)
    return overlay


def slide_info(slide, model_patch_size, mpp_model, mpp_slide, verbose=False):
    """
    Get information about the slide and calculate number of patches for grandQC model
    :param verbose: print information on the screen or not
    :param slide: WSI slide (OpenSlide)
    :param model_patch_size: patch size for grandQC model
    :param mpp_model: MPP used by the grandQC model
    :param mpp_slide: MPP of slide (we calculated the approximated MPP using information about the magnification)
    :return: patch_size - size of the patch cropped from the highest resolution of the slide
             num_patches_width - number of patches in width
             num_patches_height - number of patches in height
             width_level_0 - width of the slide at the highest resolution
             height_level_0 - height of the slide at the highest resolution
             obj_power - objective power of the slide
    """
    # Objective power
    try:
        obj_power = slide.properties["openslide.objective-power"]
    except:
        obj_power = 99

    patch_size = int(round(mpp_model / mpp_slide * model_patch_size))

    # Vendor
    vendor = slide.properties["openslide.vendor"]

    # Extract and save dimensions of level [0]
    dim_l0 = slide.level_dimensions[0]
    width_level_0 = dim_l0[0]
    height_level_0 = dim_l0[1]

    # Calculate number of patches to process
    num_patches_width = int(ceil(width_level_0 / patch_size))
    num_patches_height = int(ceil(height_level_0 / patch_size))

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
