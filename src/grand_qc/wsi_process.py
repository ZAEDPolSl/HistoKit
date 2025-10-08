from math import ceil
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

def to_tensor_x(x, **kwargs):
    """
    Converts a numpy array to torch.tensor
    :param x: numpy array
    :param kwargs:
    :return: torch.tensor
    """
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(image, preprocessing_fn, model_size):
    """
    Preprocess the image for grandQC model
    :param image: patch of the slide
    :param preprocessing_fn: function to preprocess the image
    :param model_size: size of the input patch for grandQC model
    :return: preprocessed pytorch tensor with image data
    """
    if image.size != model_size:
        image = image.resize(model_size)
        print('resized')
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x


def make_artifacts_color_map(mask, class_colors):
    """
    Make artifacts color map
    :param mask: mask with results from grandQC from 0 to 8
    :param class_colors: dictionary defining class colors
    :return: rgb - rgb image with artifacts colors
    """
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(1, len(class_colors)+1):
        idx = mask == l
        r[idx] = class_colors[l-1][0]
        g[idx] = class_colors[l-1][1]
        b[idx] = class_colors[l-1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def slide_process_single(model, tis_det, slide, num_patches_width, num_patches_height, org_patch_size, model_patch_size, colors,
                         encoder_model, encoder_weights, device, background_class, mpp_model, mpp_slide, width_level_0, height_level_0, size_mask_tissue):
    """
    Process a single slide with GrandQC model for artifacts detection. It also returns a mask with detected tisuse region of the i
    :param model: GrandQC model
    :param tis_det: mask with detected tissue region (0 - mask, 1 - background)
    :param slide: WSI slide (loaded with OpenSlide)
    :param num_patches_width: number of patches in width
    :param num_patches_height: number of patches in height
    :param org_patch_size: patch size on the highest resolution
    :param model_patch_size: patch size accepted by GrandQC model
    :param colors: color mapping for detected artifacts
    :param encoder_model: encoder model
    :param encoder_weights: weights for encoder model
    :param device: device to run on (cuda or cpu)
    :param background_class: background class (int)
    :param mpp_model: mpp used by model
    :param mpp_slide: mpp of slide
    :param width_level_0: width of the slide on the highest resolution
    :param height_level_0: height of the slide on the highest resolution
    :param size_mask_tissue: size of tissue mask (to create small thumbnail with mapped colors)
    :return: artifacts_color_map - small thumbnail with mapped colors
             end_image - final grandQC mask (with size defined by model MPP corresponding to 10x by default)
             end_image_bg - mask with detected tissue region (1 - mask, 0 - background) with the same size as end_image
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
                mask_raw[mask_raw == 7] = 8
                mask = np.where(td_patch_ != 0, background_class, mask_raw)
            else:
                mask = np.full(model_size, background_class)

            if wi == 0:
                temp_image = mask
                temp_image_bg = td_patch_
            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)
                temp_image_bg = np.concatenate((temp_image_bg, td_patch_), axis=1)
                print("processed patch")

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

    artifacts_color_map = make_artifacts_color_map(end_image, colors)
    artifacts_color_map = Image.fromarray(artifacts_color_map)
    artifacts_color_map = artifacts_color_map.resize(size_mask_tissue, Image.Resampling.NEAREST)

    return artifacts_color_map, end_image, end_image_bg



