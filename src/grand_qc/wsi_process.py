import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(image, preprocessing_fn, model_size):
    if image.size != model_size:
        image = image.resize(model_size)
        print('resized')
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x


def make_artifacts_color_map(mask, class_colors):
    """

    :param mask:
    :param class_colors:
    :return:
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

    :param model:
    :param tis_det:
    :param slide:
    :param num_patches_width:
    :param num_patches_height:
    :param org_patch_size:
    :param model_patch_size:
    :param colors:
    :param encoder_model:
    :param encoder_weights:
    :param device:
    :param background_class:
    :param mpp_model:
    :param mpp_slide:
    :param width_level_0:
    :param height_level_0:
    :param size_mask_tissue:
    :return:
    """

    model_size = (model_patch_size, model_patch_size)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_model, encoder_weights)

    for he in range(num_patches_height):
        h = he * org_patch_size + 1
        if he == 0:
            h = 0
        for wi in range(num_patches_width):
            w = wi * org_patch_size + 1
            if wi == 0:
                w = 0

            td_patch = tis_det[he * model_patch_size:(he + 1) * model_patch_size, wi * model_patch_size:(wi + 1) * model_patch_size]
            if td_patch.shape != model_size:
                # td_patch padding (incase td_patch does not equal (512,512))
                original_shape = td_patch.shape

                # Calculate padding needed
                padding = [(0, model_size[i] - original_shape[i]) for i in range(2)]

                # Apply padding
                td_patch_ = np.pad(td_patch, padding, mode='constant', constant_values=1)
            else:
                td_patch_ = td_patch

            if np.count_nonzero(td_patch == 0) > 50: #here change to check of segmentation map
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
            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)

        if he == 0:
            end_image = temp_image

        else:
            end_image = np.concatenate((end_image, temp_image), axis=0)

    # now get size of padded region (buffer) at Model MPP
    buffer_right_l = int((width_level_0 - (num_patches_width * org_patch_size)) * mpp_slide / mpp_model)
    buffer_bottom_l = int((height_level_0 - (num_patches_height * org_patch_size)) * mpp_slide / mpp_model)

    # firstly bottom
    buffer_bottom = np.full((buffer_bottom_l, end_image.shape[1]), background_class)
    temp_image = np.concatenate((end_image, buffer_bottom), axis=0)

    # now right side
    temp_image_he, temp_image_wi = temp_image.shape  # width and height
    buffer_right = np.full((temp_image_he, buffer_right_l), background_class)
    end_image = np.concatenate((temp_image, buffer_right), axis=1).astype(np.uint8)

    artifacts_color_map = make_artifacts_color_map(end_image, colors)
    artifacts_color_map = Image.fromarray(artifacts_color_map)
    artifacts_color_map = artifacts_color_map.resize(size_mask_tissue, Image.Resampling.NEAREST)

    return artifacts_color_map, end_image



