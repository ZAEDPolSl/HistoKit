from PIL import Image
from math import ceil
import numpy as np
from skimage import measure

def slide_info(slide, model_patch_size, mpp_model, mpp_slide, verbose=False):
    """
    Retrieve basic information about a whole-slide image (WSI) and compute patch grid for GrandQC model.

    This function calculates the patch size at the slide's highest resolution and
    the number of patches along width and height needed to cover the slide for processing.
    Additional metadata such as objective power, vendor, and level downsamples are also extracted.

    Parameters
    ----------
    slide : OpenSlide object
        Whole-slide image loaded via OpenSlide.
    model_patch_size : int
        Patch size expected by the GrandQC model.
    mpp_model : float
        Microns per pixel (MPP) used by the model.
    mpp_slide : float
        MPP of the slide, estimated from magnification.
    verbose : bool, default=False
        If True, prints slide information to the console.

    Returns
    -------
    patch_size : int
        Patch size at the highest resolution of the slide.
    num_patches_width : int
        Number of patches along the slide width.
    num_patches_height : int
        Number of patches along the slide height.
    width_level_0 : int
        Width of the slide at the highest resolution (level 0).
    height_level_0 : int
        Height of the slide at the highest resolution (level 0).
    obj_power : float
        Objective magnification of the slide.

    Notes
    -----
    - The patch size is scaled according to the ratio between model MPP and slide MPP.

    Examples
    --------
    >>> patch_size, num_w, num_h, width, height, obj_power = slide_info(slide, model_patch_size=512,
    ...                                                       mpp_model=0.5, mpp_slide=0.25, verbose=True)

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

    # Output
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

def get_regions_location(bg_mask):
    """
        Extract bounding boxes of connected tissue regions from a binary tissue mask.

        This function identifies connected components in the input mask and returns
        their bounding box coordinates. Each bounding box is represented as:
        ``[y_min, x_min, y_max, x_max]`` in pixel coordinates.

        Parameters
        ----------
        bg_mask : ndarray of bool or int
            2D binary mask where non-zero values indicate tissue (foreground)
            and zero indicates background.

        Returns
        -------
        list of list of int
            A list of bounding boxes, where each bounding box is represented as:
            ``[y_min, x_min, y_max, x_max]``. If no tissue regions are found,
            an empty list is returned.

        Notes
        -----
        - The mask is expected to be 2D.
        - Bounding boxes are returned in the coordinate order used by
          ``skimage.measure.regionprops``.

        Examples
        --------
        >>> mask = np.array([
        ...     [0, 1, 1, 0],
        ...     [0, 1, 1, 0],
        ...     [0, 0, 0, 0]
        ... ], dtype=bool)
        >>> get_regions_location(mask)
        [[0, 1, 2, 3]]
    """

    label_img = measure.label(bg_mask)
    props = measure.regionprops(label_img)

    bbox_list = []

    for region in props:
        bbox = region.bbox
        # y_min, x_min, y_max, x_max
        bbox_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
    return bbox_list


def load_wsi_mag(wsi, desired_mag, rescale_method=Image.LANCZOS, verbose=False, allow_upscaling=True):
    """
    Load and rescale a whole-slide image (WSI) to a desired magnification.

    This function reads the WSI at the closest available level to the desired
    magnification. If the exact magnification is unavailable, it rescales the
    image using the best level for downsample and the specified resampling method. When
    desired magnification is 5x and wsi has only magnifications 40x, 20x, 10x, 2.5x,
    the image will be rescaled from the magnification 10x to 5x. Optionally, upscaling is allowed
    when the desired magnification is higher than the highest WSI magnification.

    Parameters
    ----------
    wsi : OpenSlide object
        OpenSlide WSI object to load and rescale.
    desired_mag : float
        Desired slide magnification (e.g., 10, 20, 40).
    rescale_method : PIL.Image.Resampling or int, optional
        Resampling method used when resizing the image. Options include
        `Image.BICUBIC`, `Image.BILINEAR`, `Image.BOX`, `Image.HAMMING`,
        `Image.LANCZOS`, `Image.NEAREST`. Default is `Image.LANCZOS`.
    verbose : bool, optional
        If True, prints information about the rescaling process. Default is False.
    allow_upscaling : bool, optional
        If True, allows upscaling when the desired magnification is higher than
        the highest magnification available. Default is True.

    Returns
    -------
    region : PIL.Image.Image
        Rescaled WSI region at the desired magnification (converted to RGB).
    scale_val : float
        Scaling factor applied relative to the highest-resolution WSI level.
        For example, if the highest level is 40x and desired magnification is 10x,
        `scale_val = 40/10 = 4`.
    info : str
        Information message describing whether the desired magnification was
        available or if rescaling/upscaling was applied.
    mpp_slide : float
        Approximate microns-per-pixel (MPP) of the slide based on the highest magnification.
    ratio : list of float
        List of downsample ratios for each WSI level.

    Notes
    -----
    - If the desired magnification is available among the WSI levels, no rescaling
      is performed.
    - Rescaling is performed from the highest magnification level if the exact
      desired magnification is unavailable.
    - The function converts any RGBA images to RGB.

    Examples
    --------
    >>> region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, desired_mag=10)
    >>> print(info)
    >>> region.show()
    """

    ratio = wsi.level_downsamples
    mag_l0 = float(wsi.properties["openslide.objective-power"])
    mag_layers = [round(mag_l0 / r, 2) for r in ratio]
    mpp_slide = 10 / mag_layers[0]  # approximated slide mpp
    w0, h0 = wsi.level_dimensions[0]

    if desired_mag in mag_layers:
        info = "Desired magnification is available"
        mag_idx = mag_layers.index(desired_mag)
        w, h = wsi.level_dimensions[mag_idx]
        region = wsi.read_region((0, 0), mag_idx, (w, h))
        scale_val = ratio[mag_idx]
    elif desired_mag < mag_l0:
        info = "Desired resolution is not available, image will be rescaled from the best level for downsample."
        desired_ratio = mag_l0 / desired_mag
        level = wsi.get_best_level_for_downsample(desired_ratio)
        w, h = wsi.level_dimensions[level]
        region = wsi.read_region((0, 0), level, (w, h))
        scale_val = desired_mag / mag_layers[level]
        region = region.resize((int(w * scale_val), int(h * scale_val)), rescale_method)
    else:
        if not allow_upscaling:
            raise ValueError("The desired magnification is smaller than the highest magnification available. "
                             "The parameter allow_upscaling is set to False, so the image will not be upscaled. "
                             "If you want to upscale the image, set the parameter allow_upscaling to True. ")
        else:
            info = "Desired resolution is larger than available, image will be rescaled from the highest magnification available."
            region = wsi.read_region((0, 0), 0, (w0, h0))
            scale_val = desired_mag / mag_l0
            region = region.resize((int(w0 * scale_val), int(h0 * scale_val)), rescale_method)

    # convert RGBA to RGB
    region = region.convert("RGB")

    if verbose:
        print(info)

    return region, scale_val, info, mpp_slide, ratio


def read_region(wsi, mask_file, region_idx, desired_mag, notation="python",
                allow_list=(1, 7), resampling_method=Image.Resampling.LANCZOS):
    """
    Read a masked region from a whole-slide image (WSI) and rescale it to a desired magnification.

    This function extracts a specified region from a WSI using bounding box information
    stored in a mask file. It applies artifact filtering, rescales the region to the
    desired magnification, and converts background pixels to white.

    Parameters
    ----------
    wsi : OpenSlide object
        OpenSlide WSI object from which to read the region.
    mask_file : dict-like
        Dictionary or NumPy file containing region bounding boxes, masks, and scaling information.
    region_idx : int
        Index of the region to read from the mask file.
    desired_mag : float
        Target magnification for the output region.
    notation : {'python', 'matlab'}, optional
        Specifies whether bounding boxes use Python (0-based) or MATLAB (1-based) indexing.
        Default is "python".
    allow_list : tuple of int, optional
        Artifacts to allow in the mask. Only pixels labeled with these artifact types
        will be kept. Default is `1, 7`.
    resampling_method : PIL.Image.Resampling, optional
        Resampling method used when resizing regions (e.g., `Image.Resampling.LANCZOS`).
        Default is `Image.Resampling.LANCZOS`.

    Returns
    -------
    region_masked : ndarray of shape (H, W, 3)
        Masked and rescaled RGB region. Background pixels are set to white ([255, 255, 255]).

    Notes
    -----
    - Reads the WSI at the level closest to the desired magnification. If an exact
      level is not available, the region is rescaled using the specified resampling method.
    - Masks are resized to match the extracted region, and only allowed artifact regions
      are retained.
    - Pixels outside allowed regions are set to white for visualization.

    Examples
    --------
    >>> region = read_region(wsi, mask_file, region_idx=0, desired_mag=10)
    >>> plt.imshow(region)
    >>> plt.show()
    """

    bbox = np.array(mask_file["tiss_stats"][region_idx])

    # change matlab indexing to python indexing
    if notation == "matlab":
        bbox = bbox - 1

    mag_l0 = float(wsi.properties["openslide.objective-power"])
    desired_ratio = mag_l0 / desired_mag
    scale_val = mask_file["scale_val"] / desired_ratio
    bbox = (bbox * scale_val).astype(int)

    w0, h0 = wsi.level_dimensions[0]
    des_w, des_h = int(w0 / desired_ratio), int(h0 / desired_ratio)

    # load region defined by bbox
    if (des_w, des_h) in wsi.level_dimensions:
        level = wsi.level_dimensions.index((des_w, des_h))
        # when desired magnification is not available - read the layer with the ratio that
        # is the nearest larger ratio to the desired ratio and resize
        region = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB")
        region = region.resize((des_w, des_h), resampling_method)
        region = np.array(region)[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    else:
        # when desired ratio is available read the correct wsi layer
        level = wsi.get_best_level_for_downsample(desired_ratio)
        region = np.array(
            wsi.read_region((bbox[0], bbox[1]), level, (bbox[2] - bbox[0], bbox[3] - bbox[1])).convert("RGB"))

    # load mask with artifacts
    mask_art = mask_file["mask_art"][region_idx]
    mask = np.zeros_like(mask_art)

    # take regions from allow list
    for i in allow_list:
        mask[mask_art == i] = 1

    # Resize mask for desired resolution
    mask = np.array(Image.fromarray(mask).resize((region.shape[1], region.shape[0]), Image.Resampling.NEAREST))

    # get only areas defined by mask
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    region_masked = region * mask_rgb

    # turn bg pixels to white
    region_masked[np.all(region_masked == [0, 0, 0], axis=-1)] = [255, 255, 255]

    return region_masked