import numpy as np
import pytest
from histokit.slide.bbox import BBox


def test_init_from_list():
    bbox = BBox([10, 20, 100, 50], mag=2.5)

    assert bbox.x0 == 10.0
    assert bbox.y0 == 20.0
    assert bbox.w == 100.0
    assert bbox.h == 50.0
    assert bbox.x1 == 110.0
    assert bbox.y1 == 70.0
    assert bbox.mag == 2.5


def test_init_from_numpy_array():
    bbox = BBox(np.array([10, 20, 100, 50]), mag=2.5)

    assert bbox.xywh == (10.0, 20.0, 100.0, 50.0)
    assert bbox.xyxy == (10.0, 20.0, 110.0, 70.0)


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 0, 10],
        [0, 0, 10, 0],
        [0, 0, -1, 10],
        [0, 0, 10, -1],
    ],
)
def test_invalid_non_positive_width_or_height_raises(bbox):
    with pytest.raises(ValueError, match="positive width and height"):
        BBox(bbox)


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 10],
        [0, 0, 10, 20, 30],
        "abcd",
        [0, 0, "10", 20],
        object(),
    ],
)
def test_invalid_bbox_input_raises(bbox):
    with pytest.raises((TypeError, ValueError)):
        BBox(bbox)


@pytest.mark.parametrize(
    "mag",
    [0, -1, -2.5],
)
def test_invalid_mag_raises(mag):
    with pytest.raises(ValueError, match="mag must be positive"):
        BBox([0, 0, 10, 20], mag=mag)


@pytest.mark.parametrize(
    "mpp",
    [0, -1, -0.25],
)
def test_invalid_mpp_raises(mpp):
    with pytest.raises(ValueError, match="mpp must be positive"):
        BBox([0, 0, 10, 20], mpp=mpp)


def test_mpp_is_derived_from_mag():
    bbox = BBox([0, 0, 10, 20], mag=20.0)

    assert bbox.mag == 20.0
    assert bbox.mpp == 0.5


def test_mag_is_derived_from_mpp():
    bbox = BBox([0, 0, 10, 20], mpp=0.5)

    assert bbox.mpp == 0.5
    assert bbox.mag == 20.0


def test_mag_and_mpp_use_custom_reference_values():
    bbox = BBox(
        [0, 0, 10, 20],
        mag=40.0,
        ref_mag=20.0,
        ref_mpp=0.5,
    )

    assert bbox.mpp == 0.25


def test_area():
    bbox = BBox([10, 20, 100, 50])

    assert bbox.area() == 5000.0


def test_numpy_returns_xywh_by_default():
    bbox = BBox([10, 20, 100, 50])

    np.testing.assert_array_equal(
        bbox.numpy(),
        np.array([10.0, 20.0, 100.0, 50.0]),
    )


def test_numpy_with_int_dtype():
    bbox = BBox([10.2, 20.7, 100.4, 50.9])

    np.testing.assert_array_equal(
        bbox.numpy(dtype=int),
        np.array([10, 20, 100, 50]),
    )


def test_scale_with_factor():
    bbox = BBox([10, 20, 100, 50], mag=10.0)

    scaled = bbox.scale(factor=2.0)

    assert scaled.xywh == (20.0, 40.0, 200.0, 100.0)
    assert scaled.mag == 20.0
    assert scaled.mpp == 0.5


def test_scale_with_target_mag():
    bbox = BBox([10, 20, 100, 50], mag=10.0)

    scaled = bbox.scale(target_mag=20.0)

    assert scaled.xywh == (20.0, 40.0, 200.0, 100.0)
    assert scaled.mag == 20.0
    assert scaled.mpp == 0.5


def test_scale_with_target_mpp():
    bbox = BBox([10, 20, 100, 50], mpp=1.0)

    scaled = bbox.scale(target_mpp=0.5)

    assert scaled.xywh == (20.0, 40.0, 200.0, 100.0)
    assert scaled.mpp == 0.5
    assert scaled.mag == 20.0


def test_scale_without_factor_or_known_resolution_raises():
    bbox = BBox([10, 20, 100, 50])

    with pytest.raises(ValueError, match="Must provide either"):
        bbox.scale(target_mag=20.0)


@pytest.mark.parametrize(
    "factor",
    [0, -1, -0.5],
)
def test_scale_with_invalid_factor_raises(factor):
    bbox = BBox([10, 20, 100, 50])

    with pytest.raises(ValueError, match="Scaling factor must be positive"):
        bbox.scale(factor=factor)


def test_get_bbox_integer_covers_float_bbox():
    bbox = BBox([10.8, 15.2, 5.4, 3.9], mag=2.5)

    int_bbox = bbox.get_bbox_integer()

    assert int_bbox.xywh == (10.0, 15.0, 7.0, 5.0)
    assert int_bbox.xyxy == (10.0, 15.0, 17.0, 20.0)
    assert int_bbox.mag == 2.5


def test_get_bbox_integer_for_integer_bbox():
    bbox = BBox([10, 15, 5, 4])

    int_bbox = bbox.get_bbox_integer()

    assert int_bbox.xywh == (10.0, 15.0, 5.0, 4.0)
    assert int_bbox.xyxy == (10.0, 15.0, 15.0, 19.0)


def test_size_returns_width_height():
    bbox = BBox([10.8, 15.2, 5.4, 3.9])

    assert bbox.size == (7, 5)


def test_shape_returns_height_width():
    bbox = BBox([10.8, 15.2, 5.4, 3.9])

    assert bbox.shape == (5, 7)


def test_xywh_int():
    bbox = BBox([10.8, 15.2, 5.4, 3.9])

    assert bbox.xywh_int == (10, 15, 7, 5)


def test_xyxy_int():
    bbox = BBox([10.8, 15.2, 5.4, 3.9])

    assert bbox.xyxy_int == (10, 15, 17, 20)


def test_numpy_int():
    bbox = BBox([10.8, 15.2, 5.4, 3.9])

    np.testing.assert_array_equal(
        bbox.numpy_int(),
        np.array([10, 15, 7, 5]),
    )


def test_repr_contains_bbox_fields():
    bbox = BBox([10, 20, 100, 50], mag=2.5)

    text = repr(bbox)

    assert "BBox" in text
    assert "x0=10.0" in text
    assert "y0=20.0" in text
    assert "w=100.0" in text
    assert "h=50.0" in text
    assert "mag=2.5" in text