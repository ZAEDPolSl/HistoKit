import numpy as np
import pytest

from histokit.stain_normalisation import StainNormalizationError
from histokit.stain_normalisation import (
    rgb2od,
    od2rgb,
    normalize_matrix,
    get_concentrations,
    is_rgb_uint8,
    get_tissue_mask,
)


def test_rgb2od_returns_float32_with_same_shape():
    img = np.full((4, 5, 3), 255, dtype=np.uint8)

    od = rgb2od(img.copy())

    assert od.shape == img.shape
    assert od.dtype == np.float32
    assert np.all(od >= 0)


def test_rgb2od_handles_zero_values():
    img = np.zeros((2, 2, 3), dtype=np.uint8)

    od = rgb2od(img.copy())

    assert np.all(np.isfinite(od))
    assert np.all(od > 0)


def test_od2rgb_returns_uint8():
    od = np.ones((4, 5, 3), dtype=np.float32) * 0.5

    rgb = od2rgb(od)

    assert rgb.shape == od.shape
    assert rgb.dtype == np.uint8


def test_normalize_matrix_normalizes_rows():
    matrix = np.array(
        [
            [3.0, 4.0],
            [1.0, 0.0],
        ]
    )

    normalized = normalize_matrix(matrix)

    norms = np.linalg.norm(normalized, axis=1)

    np.testing.assert_allclose(norms, [1.0, 1.0])


def test_normalize_matrix_raises_for_zero_row():
    matrix = np.array(
        [
            [1.0, 2.0],
            [0.0, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="zero L2 norm"):
        normalize_matrix(matrix)


def test_get_concentrations_shape():
    img = np.full((3, 4, 3), 180, dtype=np.uint8)

    stain_matrix = np.array(
        [
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
        ],
        dtype=np.float32,
    )

    concentrations = get_concentrations(
        img,
        stain_matrix,
        regularizer=0.01,
    )

    assert concentrations.shape == (12, 2)


def test_is_rgb_uint8_true_for_rgb_uint8():
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    assert is_rgb_uint8(img) is True


def test_is_rgb_uint8_false_for_float_image():
    img = np.zeros((10, 10, 3), dtype=np.float32)

    assert is_rgb_uint8(img) is False


def test_is_rgb_uint8_false_for_grayscale_image():
    img = np.zeros((10, 10), dtype=np.uint8)

    assert is_rgb_uint8(img) is False


def test_get_tissue_mask_raises_for_non_uint8_image():
    img = np.zeros((10, 10, 3), dtype=np.float32)

    with pytest.raises(StainNormalizationError, match="RGB uint8"):
        get_tissue_mask(img)


def test_get_tissue_mask_returns_boolean_mask_for_colored_region():
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    img[10:40, 10:40] = [150, 60, 120]

    mask = get_tissue_mask(
        img,
        min_size=10,
    )

    assert mask.shape == img.shape[:2]
    assert mask.dtype == bool
    assert mask.sum() > 0


def test_get_tissue_mask_raises_for_empty_mask():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)

    with pytest.raises(StainNormalizationError, match="Empty mask"):
        get_tissue_mask(
            img,
            min_size=10,
        )