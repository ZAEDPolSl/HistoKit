import numpy as np
import pytest

from histokit.slide.mask_utils import split_regions, merge_regions


def test_split_and_merge_regions_binary_mask():
    mask = np.zeros((10, 12), dtype=np.uint8)

    mask[1:4, 2:5] = 1
    mask[6:9, 8:11] = 1

    masks, bboxes = split_regions(mask)

    assert len(masks) == 2
    assert bboxes.shape == (2, 4)

    merged = merge_regions(masks, bboxes, shape=mask.shape)

    np.testing.assert_array_equal(merged, mask)


def test_split_regions_returns_expected_bboxes():
    mask = np.zeros((10, 12), dtype=np.uint8)

    mask[1:4, 2:5] = 1
    mask[6:9, 8:11] = 1

    masks, bboxes = split_regions(mask)

    expected_bboxes = np.array([
        [2, 1, 3, 3],
        [8, 6, 3, 3],
    ])

    np.testing.assert_array_equal(bboxes, expected_bboxes)

    assert masks[0].shape == (3, 3)
    assert masks[1].shape == (3, 3)


def test_split_and_merge_regions_255_mask():
    mask = np.zeros((8, 8), dtype=np.uint8)

    mask[1:3, 1:3] = 255
    mask[5:7, 4:6] = 255

    masks, bboxes = split_regions(mask)

    assert len(masks) == 2
    assert all(region_mask.dtype == mask.dtype for region_mask in masks)

    merged = merge_regions(masks, bboxes, shape=mask.shape)

    np.testing.assert_array_equal(merged, mask)


def test_split_regions_diagonal_connectivity():
    mask = np.zeros((5, 5), dtype=np.uint8)

    mask[1, 1] = 1
    mask[2, 2] = 1

    masks, bboxes = split_regions(mask)

    assert len(masks) == 1

    expected_bbox = np.array([[1, 1, 2, 2]])
    np.testing.assert_array_equal(bboxes, expected_bbox)

    merged = merge_regions(masks, bboxes, shape=mask.shape)
    np.testing.assert_array_equal(merged, mask)


def test_split_regions_rejects_non_2d_mask():
    mask = np.zeros((5, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected 2D mask"):
        split_regions(mask)


def test_split_regions_empty_mask():
    mask = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        split_regions(mask)