import numpy as np
import pytest
from PIL import Image

from histokit.slide.bbox import BBox
from histokit.slide.mask import SpatialMask


def test_init_with_mag_creates_full_bbox():
    data = np.zeros((10, 20), dtype=np.uint8)

    mask = SpatialMask(data, mag=2.5)

    assert mask.data is data
    assert mask.kind == "label"
    assert mask.shape == (10, 20)
    assert mask.size == (20, 10)
    assert mask.bbox.xywh_int == (0, 0, 20, 10)
    assert mask.mag == 2.5
    assert mask.mpp == 4.0


def test_init_with_mpp_creates_full_bbox():
    data = np.zeros((10, 20), dtype=np.uint8)

    mask = SpatialMask(data, mpp=4.0)

    assert mask.bbox.xywh_int == (0, 0, 20, 10)
    assert mask.mpp == 4.0
    assert mask.mag == 2.5


def test_init_with_bbox_uses_bbox_metadata():
    data = np.zeros((10, 20), dtype=np.uint8)
    bbox = BBox([5, 7, 20, 10], mag=2.5)

    mask = SpatialMask(data, bbox=bbox)

    assert mask.bbox.xywh_int == (5, 7, 20, 10)
    assert mask.mag == 2.5
    assert mask.mpp == 4.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"bbox": BBox([0, 0, 20, 10], mag=2.5), "mag": 2.5},
        {"bbox": BBox([0, 0, 20, 10], mag=2.5), "mpp": 4.0},
        {"mag": 2.5, "mpp": 4.0},
        {"bbox": BBox([0, 0, 20, 10], mag=2.5), "mag": 2.5, "mpp": 4.0},
    ],
)
def test_init_requires_exactly_one_spatial_source(kwargs):
    data = np.zeros((10, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="Provide exactly one"):
        SpatialMask(data, **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((10,), dtype=np.uint8),
        np.zeros((2, 3, 4, 5), dtype=np.uint8),
    ],
)
def test_init_rejects_invalid_data_dimensions(data):
    with pytest.raises(ValueError, match="Expected 2D or 3D mask"):
        SpatialMask(data, mag=2.5)


def test_init_rejects_invalid_kind():
    data = np.zeros((10, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown mask kind"):
        SpatialMask(data, mag=2.5, kind="wrong")


def test_init_rejects_shape_mismatch_with_bbox():
    data = np.zeros((10, 20), dtype=np.uint8)
    bbox = BBox([0, 0, 21, 10], mag=2.5)

    with pytest.raises(ValueError, match="does not match bbox shape"):
        SpatialMask(data, bbox=bbox)


def test_resampling_method_for_label_mask():
    data = np.zeros((10, 20), dtype=np.uint8)

    mask = SpatialMask(data, mag=2.5, kind="label")

    assert mask.resampling_method == Image.Resampling.NEAREST


def test_resampling_method_for_probability_mask():
    data = np.zeros((10, 20, 3), dtype=np.float32)

    mask = SpatialMask(data, mag=2.5, kind="probability")

    assert mask.resampling_method == Image.Resampling.BILINEAR


def test_resize_to_bbox_2d_label_mask():
    data = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.uint8,
    )
    mask = SpatialMask(data, mag=10.0, kind="label")

    target_bbox = BBox([0, 0, 4, 4], mag=20.0)
    resized = mask.resize_to_bbox(target_bbox)

    assert resized.data.shape == (4, 4)
    assert resized.bbox.xywh_int == (0, 0, 4, 4)
    assert resized.kind == "label"
    assert resized.data.dtype == data.dtype
    assert set(np.unique(resized.data)).issubset({0, 1, 2})


def test_resize_to_bbox_3d_probability_mask():
    data = np.zeros((2, 2, 3), dtype=np.float32)
    data[0, 0, 0] = 1.0
    data[1, 1, 2] = 1.0

    mask = SpatialMask(data, mag=10.0, kind="probability")

    target_bbox = BBox([0, 0, 4, 4], mag=20.0)
    resized = mask.resize_to_bbox(target_bbox)

    assert resized.data.shape == (4, 4, 3)
    assert resized.bbox.xywh_int == (0, 0, 4, 4)
    assert resized.kind == "probability"
    assert resized.data.dtype == data.dtype


def test_resize_array_rejects_invalid_size():
    data = np.zeros((10, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid output size"):
        SpatialMask._resize_array(
            data,
            size=(0, 10),
            resample=Image.Resampling.NEAREST,
        )


def test_resize_array_rejects_invalid_dimensions():
    data = np.zeros((2, 3, 4, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected 2D or 3D mask"):
        SpatialMask._resize_array(
            data,
            size=(10, 10),
            resample=Image.Resampling.NEAREST,
        )


def test_scale_with_factor():
    data = np.zeros((10, 20), dtype=np.uint8)
    mask = SpatialMask(data, mag=10.0)

    scaled = mask.scale(factor=2.0)

    assert scaled.data.shape == (20, 40)
    assert scaled.bbox.xywh_int == (0, 0, 40, 20)
    assert scaled.mag == 20.0
    assert scaled.mpp == 0.5


def test_scale_with_target_mag():
    data = np.zeros((10, 20), dtype=np.uint8)
    mask = SpatialMask(data, mag=10.0)

    scaled = mask.scale(target_mag=20.0)

    assert scaled.data.shape == (20, 40)
    assert scaled.bbox.xywh_int == (0, 0, 40, 20)
    assert scaled.mag == 20.0
    assert scaled.mpp == 0.5


def test_scale_with_target_mpp():
    data = np.zeros((10, 20), dtype=np.uint8)
    mask = SpatialMask(data, mpp=1.0)

    scaled = mask.scale(target_mpp=0.5)

    assert scaled.data.shape == (20, 40)
    assert scaled.bbox.xywh_int == (0, 0, 40, 20)
    assert scaled.mpp == 0.5
    assert scaled.mag == 20.0


def test_split_regions_returns_connected_components_with_bboxes():
    data = np.zeros((8, 10), dtype=np.uint8)

    data[1:3, 2:5] = 1
    data[5:7, 7:9] = 2

    mask = SpatialMask(data, mag=2.5)
    regions = mask.split_regions()

    assert len(regions) == 2

    bboxes = sorted(region.bbox.xywh_int for region in regions)
    assert bboxes == [
        (2, 1, 3, 2),
        (7, 5, 2, 2),
    ]

    region_shapes = sorted(region.data.shape for region in regions)
    assert region_shapes == [
        (2, 2),
        (2, 3),
    ]

    assert all(region.kind == "label" for region in regions)
    assert all(region.mag == 2.5 for region in regions)


def test_split_regions_with_offset_bbox():
    data = np.zeros((8, 10), dtype=np.uint8)
    data[1:3, 2:5] = 1

    bbox = BBox([100, 200, 10, 8], mag=2.5)
    mask = SpatialMask(data, bbox=bbox)

    regions = mask.split_regions()

    assert len(regions) == 1
    assert regions[0].bbox.xywh_int == (102, 201, 3, 2)


def test_split_regions_respects_min_area():
    data = np.zeros((8, 10), dtype=np.uint8)

    data[1:3, 2:5] = 1  # area 6
    data[5:6, 7:8] = 2  # area 1

    mask = SpatialMask(data, mag=2.5)
    regions = mask.split_regions(min_area=2)

    assert len(regions) == 1
    assert regions[0].bbox.xywh_int == (2, 1, 3, 2)


def test_split_regions_rejects_probability_mask():
    data = np.zeros((8, 10), dtype=np.float32)

    mask = SpatialMask(data, mag=2.5, kind="probability")

    with pytest.raises(ValueError, match="kind='label'"):
        mask.split_regions()


def test_split_regions_rejects_3d_mask():
    data = np.zeros((8, 10, 3), dtype=np.uint8)

    mask = SpatialMask(data, mag=2.5, kind="label")

    with pytest.raises(ValueError, match="Expected 2D mask"):
        mask.split_regions()


def test_merge_regions_2d_reconstructs_mask():
    data = np.zeros((8, 10), dtype=np.uint8)

    data[1:3, 2:5] = 1
    data[5:7, 7:9] = 2

    mask = SpatialMask(data, mag=2.5)
    regions = mask.split_regions()

    merged = SpatialMask.merge_regions(regions, shape=data.shape)

    np.testing.assert_array_equal(merged.data, data)
    assert merged.bbox.xywh_int == (0, 0, 10, 8)
    assert merged.kind == "label"
    assert merged.mag == 2.5


def test_merge_regions_3d_reconstructs_nonzero_regions():
    data = np.zeros((8, 10, 2), dtype=np.float32)

    data[1:3, 2:5, 0] = 0.5
    data[5:7, 7:9, 1] = 0.8

    region_1 = SpatialMask(
        data[1:3, 2:5].copy(),
        bbox=BBox([2, 1, 3, 2], mag=2.5),
        kind="probability",
    )
    region_2 = SpatialMask(
        data[5:7, 7:9].copy(),
        bbox=BBox([7, 5, 2, 2], mag=2.5),
        kind="probability",
    )

    merged = SpatialMask.merge_regions([region_1, region_2], shape=data.shape[:2])

    np.testing.assert_array_equal(merged.data, data)
    assert merged.data.shape == (8, 10, 2)
    assert merged.kind == "probability"
    assert merged.bbox.xywh_int == (0, 0, 10, 8)


def test_merge_regions_rejects_invalid_shape_argument():
    data = np.zeros((4, 5), dtype=np.uint8)
    region = SpatialMask(data, mag=2.5)

    with pytest.raises(ValueError, match="shape must be a 2-element tuple"):
        SpatialMask.merge_regions([region], shape=(4, 5, 1))


def test_merge_regions_rejects_empty_regions():
    with pytest.raises(IndexError):
        SpatialMask.merge_regions([], shape=(10, 20))



def test_to_parts_returns_data_and_integer_bbox():
    data = np.zeros((10, 20), dtype=np.uint8)
    bbox = BBox([5, 7, 20, 10], mag=2.5)

    mask = SpatialMask(data, bbox=bbox)

    out_data, out_bbox = mask.to_parts()

    assert out_data is data
    np.testing.assert_array_equal(
        out_bbox,
        np.array([5, 7, 20, 10]),
    )


def test_parts_from_regions_returns_old_style_masks_and_bboxes():
    data = np.zeros((8, 10), dtype=np.uint8)
    data[1:3, 2:5] = 1
    data[5:7, 7:9] = 2

    mask = SpatialMask(data, mag=2.5)
    regions = mask.split_regions()

    masks, bboxes = SpatialMask.parts_from_regions(regions)

    assert len(masks) == 2
    assert bboxes.shape == (2, 4)
    assert bboxes.dtype == int

    sorted_bboxes = sorted(tuple(row) for row in bboxes)
    assert sorted_bboxes == [
        (2, 1, 3, 2),
        (7, 5, 2, 2),
    ]


def test_parts_from_empty_regions():
    masks, bboxes = SpatialMask.parts_from_regions([])

    assert masks == []
    assert bboxes.shape == (0, 4)
    assert bboxes.dtype == int


def test_repr_contains_core_fields():
    data = np.zeros((10, 20), dtype=np.uint8)
    mask = SpatialMask(data, mag=2.5)

    text = repr(mask)

    assert "SpatialMask" in text
    assert "shape=(10, 20)" in text
    assert "kind=label" in text


def test_binarize_binary_01_mask():
    data = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    mask = SpatialMask(data, mag=2.5)

    returned = mask.binarize()

    assert returned is mask
    assert mask.data.dtype == bool

    np.testing.assert_array_equal(
        mask.data,
        np.array(
            [
                [False, True, False],
                [True, False, True],
            ],
            dtype=bool,
        ),
    )


def test_binarize_binary_0255_uint8_mask():
    data = np.array(
        [
            [0, 255, 0],
            [255, 0, 255],
        ],
        dtype=np.uint8,
    )

    mask = SpatialMask(data, mag=2.5)

    mask.binarize()

    assert mask.data.dtype == bool

    np.testing.assert_array_equal(
        mask.data,
        np.array(
            [
                [False, True, False],
                [True, False, True],
            ],
            dtype=bool,
        ),
    )


def test_binarize_bool_mask_returns_self_without_change():
    data = np.array(
        [
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )

    mask = SpatialMask(data, mag=2.5)

    returned = mask.binarize()

    assert returned is mask
    assert mask.data is data

    np.testing.assert_array_equal(mask.data, data)


def test_binarize_multiclass_mask_with_single_keep_value_warns():
    data = np.array(
        [
            [0, 1, 2],
            [3, 1, 0],
        ],
        dtype=np.uint8,
    )

    mask = SpatialMask(data, mag=2.5)

    with pytest.warns(UserWarning, match="mask is not bool"):
        mask.binarize(keep=1)

    assert mask.data.dtype == bool

    np.testing.assert_array_equal(
        mask.data,
        np.array(
            [
                [False, True, False],
                [False, True, False],
            ],
            dtype=bool,
        ),
    )


def test_binarize_multiclass_mask_with_multiple_keep_values_warns():
    data = np.array(
        [
            [0, 1, 2],
            [3, 1, 0],
        ],
        dtype=np.uint8,
    )

    mask = SpatialMask(data, mag=2.5)

    with pytest.warns(UserWarning, match="values in"):
        mask.binarize(keep=(1, 3))

    assert mask.data.dtype == bool

    np.testing.assert_array_equal(
        mask.data,
        np.array(
            [
                [False, True, False],
                [True, True, False],
            ],
            dtype=bool,
        ),
    )


def test_binarize_rejects_3d_mask():
    data = np.zeros((4, 5, 2), dtype=np.uint8)

    mask = SpatialMask(data, mag=2.5)

    with pytest.raises(ValueError, match="mask must be 2-dimensional"):
        mask.binarize()


def test_threshold_2d_float_mask():
    data = np.array(
        [
            [0.1, 0.5, 0.6],
            [0.9, 0.2, 0.7],
        ],
        dtype=np.float32,
    )

    mask = SpatialMask(data, mag=2.5, kind="probability")

    returned = mask.threshold(thr=0.5)

    assert returned is mask
    assert mask.data.dtype == bool

    np.testing.assert_array_equal(
        mask.data,
        np.array(
            [
                [False, False, True],
                [True, False, True],
            ],
            dtype=bool,
        ),
    )


def test_threshold_3d_float_mask():
    data = np.array(
        [
            [[0.1, 0.7], [0.6, 0.2]],
            [[0.9, 0.1], [0.4, 0.8]],
        ],
        dtype=np.float32,
    )

    mask = SpatialMask(data, mag=2.5, kind="probability")

    mask.threshold(thr=0.5)

    assert mask.data.dtype == bool
    assert mask.data.shape == (2, 2, 2)

    np.testing.assert_array_equal(
        mask.data,
        np.array(
            [
                [[False, True], [True, False]],
                [[True, False], [False, True]],
            ],
            dtype=bool,
        ),
    )


def test_threshold_bool_mask_returns_self_without_change():
    data = np.array(
        [
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )

    mask = SpatialMask(data, mag=2.5)

    returned = mask.threshold(thr=0.5)

    assert returned is mask
    assert mask.data is data

    np.testing.assert_array_equal(mask.data, data)


def test_threshold_uint8_mask():
    data = np.array(
        [
            [0, 100, 200],
            [255, 10, 127],
        ],
        dtype=np.uint8,
    )

    mask = SpatialMask(data, mag=2.5)

    mask.threshold(thr=127)

    assert mask.data.dtype == bool

    np.testing.assert_array_equal(
        mask.data,
        np.array(
            [
                [False, False, True],
                [True, False, False],
            ],
            dtype=bool,
        ),
    )


def test_threshold_rejects_invalid_dimensions():
    data = np.zeros((2, 3, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 2D or 3D mask"):
        SpatialMask(data, mag=2.5)