from math import ceil
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from .artifacts import Artifact
from ..grand_qc.visualisation import make_artifacts_color_map


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

def to_tensor_x(x, **kwargs):
    """
    Convert an RGB image from a NumPy array to a PyTorch tensor.

    This function transposes the image from HWC (height, width, channels) format
    to CHW (channels, height, width) format and casts it to `float32`, preparing
    it for input into PyTorch models.

    Parameters
    ----------
    x : ndarray of shape (H, W, C)
        RGB image as a NumPy array.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    tensor : ndarray of dtype float32, shape (C, H, W)
        Transposed and type-cast tensor suitable for PyTorch.

    Notes
    -----
    - The function does not normalize pixel values (e.g., to [0,1]). This should
      be done separately if required by the model.
    - This is a lightweight function and does not create a true `torch.Tensor`.
      To convert to a PyTorch tensor, use `torch.from_numpy(to_tensor_x(x))`.

    Examples
    --------
    >>> x_tensor = to_tensor_x(image_np)
    >>> print(x_tensor.shape)  # (3, H, W)

    References
    ----------
    This function is taken from \ :footcite:p:`Weng2024`

    .. footbibliography::
    """
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(image, preprocessing_fn, model_size):
    """
    Preprocess a slide patch for input into the GrandQC model.

    This function resizes the input patch to the required model size, applies
    a custom preprocessing function (e.g., normalization or augmentation), and
    converts the result into a PyTorch tensor.

    Parameters
    ----------
    image : PIL.Image.Image
        Image patch of the slide to preprocess.
    preprocessing_fn : callable
        Function that applies preprocessing to the image (e.g., normalization,
        color adjustments). Should accept a NumPy array and return a NumPy array.
    model_size : tuple of int (width, height)
        Target size for the model input. If the input image does not match this
        size, it will be resized.

    Returns
    -------
    x : ndarray of shape (C, H, W), dtype float32
        Preprocessed image as a PyTorch tensor (CHW format).

    Notes
    -----
    - Uses :func:`to_tensor_x` to convert the NumPy array into CHW format and
      float32 dtype.
    - The function prints 'resized' if the image was resized to match the model input.

    Examples
    --------
    >>> x_tensor = get_preprocessing(patch, preprocessing_fn, model_size=(224, 224))
    >>> print(x_tensor.shape)  # (3, 224, 224)

    References
    ----------
    This function is taken from \ :footcite:p:`Weng2024`
    """
    if image.size != model_size:
        image = image.resize(model_size)
        print('resized')
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x


def slide_process_single(model, tis_det, slide, num_patches_width, num_patches_height, org_patch_size, model_patch_size,
                         encoder_model, encoder_weights, device,  mpp_model, mpp_slide, width_level_0, height_level_0, size_mask_tissue, background_class = 0):
    """
    Process a single whole-slide image (WSI) for artifact detection using the GrandQC model.

    This function iterates over tissue patches in the WSI, applies the GrandQC model
    to detect artifacts, and returns both the artifact mask and a small RGB thumbnail
    for visualization. The output mask is resized to match the model MPP (microns per pixel), corresponding to 10x magnification.

    Parameters
    ----------
    model : torch.nn.Module
        GrandQC model for artifact detection.
    tis_det : ndarray of shape (H_model, W_model)
        Binary mask of detected tissue regions (0 - tissue, 1 - background).
    slide : OpenSlide object
        Whole-slide image (WSI) loaded via OpenSlide.
    num_patches_width : int
        Number of patches along the slide width.
    num_patches_height : int
        Number of patches along the slide height.
    org_patch_size : int
        Patch size at the original highest resolution.
    model_patch_size : int
        Patch size accepted by the GrandQC model.
    encoder_model : str
        Name of the encoder used for the GrandQC model (used for preprocessing).
    encoder_weights : str
        Weights of the encoder used for preprocessing.
    device : str or torch.device
        Device to run inference on (e.g., 'cpu' or 'cuda').
    background_class : int
        Class label representing the background.
    mpp_model : float
        Microns-per-pixel used by the model.
    mpp_slide : float
        Microns-per-pixel of the original slide.
    width_level_0 : int
        Width of the slide at the highest resolution.
    height_level_0 : int
        Height of the slide at the highest resolution.
    size_mask_tissue : tuple of int
        Size of the thumbnail to generate with mapped artifact colors.

    Returns
    -------
    artifacts_color_map : PIL.Image.Image
        Small RGB thumbnail showing artifact classes mapped to their respective colors.
    end_image : ndarray of shape (H_rescaled, W_rescaled), dtype uint8
        Final GrandQC mask resized to model MPP, where each pixel represents the predicted artifact class.
    end_image_bg : ndarray of shape (H_rescaled, W_rescaled), dtype uint8
        Binary tissue mask at the same size as `end_image` (1 - tissue, 0 - background).

    Notes
    -----
    - Patches are padded if they do not match the model patch size.
    - The `artifacts_color_map` can be used for quick visualization of detected artifacts.

    Examples
    --------
    >>> artifacts_map, mask, mask_bg = slide_process_single(model, tis_det, slide, num_patches_width=20,
    ...                                                     num_patches_height=15, org_patch_size=512,
    ...                                                     model_patch_size=512,
    ...                                                     encoder_model='resnet34', encoder_weights='imagenet',
    ...                                                     device='cuda',
    ...                                                     mpp_model=0.5, mpp_slide=0.25,
    ...                                                     width_level_0=10000, height_level_0=8000,
    ...                                                     size_mask_tissue=(512, 512))
    >>> artifacts_map.show()

    References
    ----------
    This function is based on \ :footcite:p:`Weng2024`
    """

    model_size = (model_patch_size, model_patch_size)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_model, encoder_weights)

    for he in range(num_patches_height):
        h = he * org_patch_size

        if h > 0:
            h += 1

        for wi in range(num_patches_width):
            w = wi * org_patch_size

            if w > 0:
                w += 1

            td_patch = tis_det[he * model_patch_size:(he + 1) * model_patch_size, wi * model_patch_size:(wi + 1) * model_patch_size]

            if td_patch.shape != model_size:
                # td_patch padding (incase td_patch does not equal (512,512))
                original_shape = td_patch.shape

                # Calculate padding needed
                padding = [(0, model_size[i] - original_shape[i]) for i in range(2)]

                # Apply padding, fill with background class
                td_patch_ = np.pad(td_patch, padding, mode='constant', constant_values=1)
            else:
                td_patch_ = td_patch

            if np.count_nonzero(td_patch == 0) > 0:

                # Generate patch
                work_patch = slide.read_region((w, h), 0, (org_patch_size, org_patch_size))
                work_patch = work_patch.convert('RGB')

                # Resize to model patch size
                work_patch = work_patch.resize(model_size, Image.Resampling.LANCZOS)

                image_pre = get_preprocessing(work_patch, preprocessing_fn, model_size)
                x_tensor = torch.from_numpy(image_pre).to(device).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = (predictions.squeeze().cpu().numpy())

                mask_raw = np.argmax(predictions, axis=0).astype('int8')

                # add background predicted during artifacts prediction
                mask = np.where(td_patch_ != 0, Artifact.BG_THR.value, mask_raw)
            else:
                mask = np.full(model_size, Artifact.BG_THR.value)

            if wi == 0:
                temp_image = mask
                temp_image_bg = td_patch_
            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)
                temp_image_bg = np.concatenate((temp_image_bg, td_patch_), axis=1)

        if he == 0:
            end_image = temp_image
            end_image_bg = temp_image_bg
        else:
            end_image = np.concatenate((end_image, temp_image), axis=0)
            end_image_bg = np.concatenate((end_image_bg, temp_image_bg), axis=0)

    h_whole = int(height_level_0*mpp_slide/mpp_model)
    w_whole = int(width_level_0*mpp_slide/mpp_model)
    end_image = end_image[0:h_whole, 0:w_whole].astype(np.uint8)
    end_image_bg = 1 - end_image_bg[0:h_whole, 0:w_whole].astype(np.uint8)

    artifacts_color_map = make_artifacts_color_map(end_image)
    artifacts_color_map = Image.fromarray(artifacts_color_map)
    artifacts_color_map = artifacts_color_map.resize(size_mask_tissue, Image.Resampling.NEAREST)

    return artifacts_color_map, end_image, end_image_bg



