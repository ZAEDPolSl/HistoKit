import pytest
import numpy as np
from src.histo_kit.slide.bbox import BBox, BBoxMode


def test_bbox_init_xy_and_wh():
    # x1/y1
    bbox1 = BBox(10, 20, x1=50, y1=60)
    assert bbox1.w == 40
    assert bbox1.h == 40
    assert bbox1.as_tuple(mode=BBoxMode.XY) == (10, 20, 50, 60)
    assert bbox1.area() == 1600

    # w/h
    bbox2 = BBox(10, 20, w=40, h=40)
    assert bbox2.x1 == 50
    assert bbox2.y1 == 60
    assert bbox2.area() == 1600

    with pytest.raises(ValueError):
        BBox(0, 0, x1=10, w=5, y1=10)


    with pytest.raises(ValueError):
        BBox(0, 0, h=5)

def test_mag_mpp_calculation():
    bbox = BBox(0, 0, w=10, h=10, mag=20)
    assert bbox.mag == 20
    assert bbox.mpp == round(BBox.REF_MAG * BBox.REF_MPP / 20, 4)

    bbox2 = BBox(0, 0, w=10, h=10, mpp=0.5)
    assert bbox2.mpp == 0.5
    assert bbox2.mag == round(BBox.REF_MAG * BBox.REF_MPP / 0.5, 2)

def test_center():
    bbox = BBox(0, 0, w=10, h=20)
    cx, cy = bbox.center
    assert cx == 5
    assert cy == 10

def test_scale_factor():
    bbox = BBox(10, 20, w=40, h=60, mag=20, mpp=1)
    scaled = bbox.scale(factor=0.5)
    assert scaled.x0 == round(10 * 0.5)
    assert scaled.y0 == round(20 * 0.5)
    assert scaled.w == round(40 * 0.5)
    assert scaled.h == round(60 * 0.5)
    assert scaled.mag == 20 * 0.5
    assert scaled.mpp == 1 / 0.5

def test_scale_mag_and_mpp():
    bbox = BBox(0, 0, w=10, h=10, mag=20, mpp=1)
    scaled_mag = bbox.scale(mag=10)
    assert scaled_mag.mag == 10
    assert scaled_mag.mpp == 2

    scaled_mpp = bbox.scale(mpp=2)
    assert scaled_mpp.mpp == 2
    assert scaled_mpp.mag == 10

def test_as_tuple_and_array():
    bbox = BBox(10, 20, w=40, h=50)
    assert bbox.as_tuple(BBoxMode.XY) == (10, 20, 50, 70)
    assert bbox.as_tuple(BBoxMode.WH) == (10, 20, 40, 50)

    arr_xy = bbox.as_array(BBoxMode.XY)
    assert isinstance(arr_xy, np.ndarray)
    assert np.array_equal(arr_xy, np.array([10, 20, 50, 70]))

    arr_wh = bbox.as_array(BBoxMode.WH)
    assert np.array_equal(arr_wh, np.array([10, 20, 40, 50]))

def test_area():
    bbox = BBox(0, 0, w=3, h=5)
    assert bbox.area() == 15

def test_normalize_bbox_object_no_change():
    bbox = BBox(10, 20, w=30, h=40, mag=10, mpp=1)
    normalized = BBox.normalize(bbox)
    assert normalized is bbox

def test_normalize_bbox_object_with_new_params():
    bbox = BBox(10, 20, w=30, h=40, mag=10, mpp=1)
    normalized = BBox.normalize(bbox, mag=20, mpp=0.5, ref_mag=15, ref_mpp=1.2)
    assert normalized is not bbox
    assert normalized.mag == 20
    assert normalized.mpp == 0.5
    assert normalized.ref_mag == 15
    assert normalized.ref_mpp == 1.2
    assert normalized.x0 == bbox.x0 and normalized.y0 == bbox.y0
    assert normalized.w == bbox.w and normalized.h == bbox.h

@pytest.mark.parametrize("seq_type", [list, tuple, np.array])
def test_normalize_bbox_sequence(seq_type):
    arr = seq_type([10, 20, 30, 40])
    normalized = BBox.normalize(arr, mode=BBoxMode.WH, mag=5, mpp=0.5)
    assert isinstance(normalized, BBox)
    assert normalized.x0 == 10
    assert normalized.y0 == 20
    assert normalized.w == 30
    assert normalized.h == 40
    assert normalized.mag == 5
    assert normalized.mpp == 0.5

@pytest.mark.parametrize("seq_type", [list, tuple, np.array])
def test_normalize_bbox_sequence_mag(seq_type):
    arr = seq_type([10, 20, 30, 40])
    normalized = BBox.normalize(arr, mode=BBoxMode.WH, mag=10)
    assert isinstance(normalized, BBox)
    assert normalized.x0 == 10
    assert normalized.y0 == 20
    assert normalized.w == 30
    assert normalized.h == 40
    assert normalized.mag == 10
    assert normalized.mpp == 1

@pytest.mark.parametrize("seq_type", [list, tuple, np.array])
def test_normalize_bbox_sequence_mag(seq_type):
    arr = seq_type([10, 20, 30, 40])
    normalized = BBox.normalize(arr, mode=BBoxMode.WH, mpp=1)
    assert isinstance(normalized, BBox)
    assert normalized.x0 == 10
    assert normalized.y0 == 20
    assert normalized.w == 30
    assert normalized.h == 40
    assert normalized.mag == 10
    assert normalized.mpp == 1

def test_normalize_invalid_sequence_length():
    with pytest.raises(ValueError):
        BBox.normalize([10, 20, 30])

def test_normalize_invalid_type():
    with pytest.raises(TypeError):
        BBox.normalize("Not a bbox")

@pytest.mark.parametrize(
    "bbox_input, bbox_mode, expected_count",
    [   (None, BBoxMode.WH, 0),
        (BBox(0, 0, w=10, h=10), BBoxMode.WH, 1),
        ([(0, 0, 10, 10), (20, 20, 5, 5)], BBoxMode.WH, 2),
        ([(0, 0, 10, 10)], BBoxMode.XY, 1),
        (np.array([0, 0, 10, 10]), BBoxMode.WH, 1),
        (np.array([[0, 0, 10, 10], [20, 20, 5, 5]]), BBoxMode.WH, 2),
        ([BBox(1, 1, w=5, h=5), BBox(2, 2, w=3, h=3)], BBoxMode.WH, 2),
    ]
)
def test_parse_bbox_list(bbox_input, bbox_mode, expected_count):

    result = BBox.parse_bbox_list(
        bbox=bbox_input,
        bbox_mode=bbox_mode
    )

    assert isinstance(result, list)
    assert all(isinstance(b, BBox) for b in result)
    assert len(result) == expected_count
