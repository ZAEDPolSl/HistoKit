import numpy as np
import pytest
from PIL import ImageChops
from PIL import Image
from histokit.slide.bbox import BBox, BBoxMode
from histokit.slide.slide import Slide
from tests.mocks.mock_wsi_backend import MockWSIBackend

@pytest.fixture
def slide():
    backend = MockWSIBackend()
    s = Slide.__new__(Slide)
    s._backend = backend
    s._ref_mag = 10
    s._ref_mpp = 1
    s._mag = backend.mag
    s._mpp = backend.mpp
    s.rescale_method = Image.Resampling.LANCZOS
    return s

def test_invalid_level_mag_mpp(slide):
    bbox = BBox(0, 0, w=10, h=10)
    with pytest.raises(ValueError):
        slide.read_region(bbox)

@pytest.mark.parametrize(
    "bbox, level, expected_center_pixel, bbox_mode",
    [
        ([1600, 1600, 800, 800], 0, (400, 400), BBoxMode.WH),
        ([800, 800, 400, 400], 1, (200, 200), BBoxMode.WH),
        ([400, 400, 200, 200], 2, (100, 100), BBoxMode.WH),

        (BBox(1600, 1600, w=800, h=800), 0, (400, 400), None),
        (BBox(800, 800, w=400, h=400), 1, (200, 200),  None),
        (BBox(400, 400, w=200, h=200), 2, (100, 100), None),

        (BBox(1600, 1600, w=800, h=800, mag=10), 0, (400, 400), None),
        (BBox(800, 800, w=400, h=400, mpp=3), 1, (200, 200),  None),
        (BBox(400, 400, w=200, h=200, mag=40), 2, (100, 100), None),

        (BBox(1600, 1600, x1=1600+800, y1=1600+800), 0, (400, 400), None),
        (BBox(800, 800, x1=800+400, y1=800+400), 1, (200, 200), None),
        (BBox(400, 400, x1=400+200, y1=400+200), 2, (100, 100), None),

        (np.array([1600, 1600, 1600+800, 1600+800]), 0, (400, 400), BBoxMode.XY),
        (np.array([800, 800, 800+400, 800+400]), 1, (200, 200), BBoxMode.XY),
        (np.array([400, 400, 400+200, 400+200]), 2, (100, 100), BBoxMode.XY),
    ],
)
def test_circle_read_region_level(bbox, level, expected_center_pixel, slide, bbox_mode):
    img = slide.read_region(bbox, level=level)
    pixel = img.getpixel(expected_center_pixel)[:3]
    #img.show()
    assert pixel == (255, 0, 0)

@pytest.mark.parametrize(
    "bbox, mag, expected_center_pixel, bbox_mode",
    [
        ([3200, 3200, 1600, 1600], 40, (800, 800), BBoxMode.WH),
        ([1600, 1600, 800, 800], 20, (400, 400), BBoxMode.WH),
        ([800, 800, 400, 400], 10, (200, 200), BBoxMode.WH),
        ([400, 400, 200, 200], 5, (100, 100), BBoxMode.WH),
        ([200, 200, 100, 100], 2.5, (50, 50), BBoxMode.WH),

        (BBox(3200, 3200, w=1600, h=1600), 40, (800, 800), None),
        (BBox(1600, 1600, w=800, h=800), 20, (400, 400), None),
        (BBox(800, 800, w=400, h=400), 10, (200, 200),  None),
        (BBox(400, 400, w=200, h=200), 5, (100, 100), None),
        (BBox(200, 200, w=100, h=100), 2.5, (50, 50), None),

        (BBox(3200, 3200, w=1600, h=1600, mag=40), 40, (800, 800), None),
        (BBox(1600, 1600, w=800, h=800, mag=10), 20, (400, 400), None),
        (BBox(800, 800, w=400, h=400, mpp=3), 10, (200, 200),  None),
        (BBox(400, 400, w=200, h=200, mag=40), 5, (100, 100), None),
        (BBox(200, 200, w=100, h=100, mag=40), 2.5, (50, 50), None),

        (BBox(3200, 3200, w=3200+1600, h=3200+1600), 40, (800, 800), None),
        (BBox(1600, 1600, x1=1600+800, y1=1600+800), 20, (400, 400), None),
        (BBox(800, 800, x1=800+400, y1=800+400), 10, (200, 200), None),
        (BBox(400, 400, x1=400+200, y1=400+200), 5, (100, 100), None),
        (BBox(200, 200, x1=200+100, y1=200+100), 2.5, (50, 50), None),

        (np.array([3200, 3200, 3200+1600, 3200+1600]), 40, (800, 800), BBoxMode.XY),
        (np.array([1600, 1600, 1600+800, 1600+800]), 20, (400, 400), BBoxMode.XY),
        (np.array([800, 800, 800+400, 800+400]), 10, (200, 200), BBoxMode.XY),
        (np.array([400, 400, 400+200, 400+200]), 5, (100, 100), BBoxMode.XY),
        (np.array([200, 200, 200+100, 200+100]), 2.5, (50, 50), BBoxMode.XY),
    ],
)
def test_circle_read_region_mag(bbox, mag, expected_center_pixel, slide, bbox_mode):
    img = slide.read_region(bbox, mag=mag)
    pixel = img.getpixel(expected_center_pixel)[:3]
    #img.show()
    assert pixel[0]>=180 # Allow some blurring due to resampling

@pytest.mark.parametrize(
    "bbox, mpp, expected_center_pixel, bbox_mode",
    [
        ([3200, 3200, 1600, 1600], 0.25, (800, 800), BBoxMode.WH),
        ([1600, 1600, 800, 800], 0.5, (400, 400), BBoxMode.WH),
        ([800, 800, 400, 400], 1, (200, 200), BBoxMode.WH),
        ([400, 400, 200, 200], 2, (100, 100), BBoxMode.WH),
        ([200, 200, 100, 100], 4, (50, 50), BBoxMode.WH),

        (BBox(3200, 3200, w=1600, h=1600), 0.25, (800, 800), None),
        (BBox(1600, 1600, w=800, h=800), 0.5, (400, 400), None),
        (BBox(800, 800, w=400, h=400), 1, (200, 200),  None),
        (BBox(400, 400, w=200, h=200), 2, (100, 100), None),
        (BBox(200, 200, w=100, h=100), 4, (50, 50), None),

        (BBox(3200, 3200, w=1600, h=1600, mag=40), 0.25, (800, 800), None),
        (BBox(1600, 1600, w=800, h=800, mag=10), 0.5, (400, 400), None),
        (BBox(800, 800, w=400, h=400, mpp=3), 1, (200, 200),  None),
        (BBox(400, 400, w=200, h=200, mag=40), 2, (100, 100), None),
        (BBox(200, 200, w=100, h=100, mag=40), 4, (50, 50), None),

        (BBox(3200, 3200, w=3200+1600, h=3200+1600), 0.25, (800, 800), None),
        (BBox(1600, 1600, x1=1600+800, y1=1600+800), 0.5, (400, 400), None),
        (BBox(800, 800, x1=800+400, y1=800+400), 1, (200, 200), None),
        (BBox(400, 400, x1=400+200, y1=400+200), 2, (100, 100), None),
        (BBox(200, 200, x1=200+100, y1=200+100), 4, (50, 50), None),

        (np.array([3200, 3200, 3200+1600, 3200+1600]), 0.25, (800, 800), BBoxMode.XY),
        (np.array([1600, 1600, 1600+800, 1600+800]), 0.5, (400, 400), BBoxMode.XY),
        (np.array([800, 800, 800+400, 800+400]), 1, (200, 200), BBoxMode.XY),
        (np.array([400, 400, 400+200, 400+200]), 2, (100, 100), BBoxMode.XY),
        (np.array([200, 200, 200+100, 200+100]), 4, (50, 50), BBoxMode.XY),
    ],
)
def test_circle_read_region_mag(bbox, mpp, expected_center_pixel, slide, bbox_mode):
    img = slide.read_region(bbox, mpp=mpp)
    pixel = img.getpixel(expected_center_pixel)[:3]
    #img.show()
    assert pixel[0]>=180 # Allow some blurring due to resampling


@pytest.mark.parametrize(
    "bbox, mpp,mag, expected_center_pixel, bbox_mode, mpp_extract",
    [
        (BBox(3200, 3200, w=1600, h=1600, mpp = 0.25), 0.25, 40, (200, 200), None, 1),
        (BBox(1600, 1600, w=800, h=800, mpp = 0.5), 0.5, 20, (200, 200), None, 1),
        (BBox(800, 800, w=400, h=400, mpp = 1), 1, 10, (200, 200),  None, 1),
        (BBox(400, 400, w=200, h=200, mpp = 2), 2, 5, (200, 200), None, 1),
        (BBox(200, 200, w=100, h=100, mpp = 4), 4, 2.5, (200, 200), None, 1),

        (BBox(3200, 3200, w=1600, h=1600, mag = 40), None, None, (200, 200), None, 1),
        (BBox(1600, 1600, w=800, h=800, mag = 20), 0.5, None, (200, 200), None, 1),
        (BBox(400, 400, w=200, h=200, mag = 5), None, None, (200, 200), None, 1),
        (BBox(200, 200, w=100, h=100, mag = 2.5), None, 40, (200, 200), None, 1),

        (np.array([3200, 3200, 3200+1600, 3200+1600]), 0.25, 40,(200, 200), BBoxMode.XY,1),
        (np.array([1600, 1600, 1600+800, 1600+800]), 0.5, 20,(200, 200), BBoxMode.XY, 1),
        (np.array([800, 800, 800+400, 800+400]), 1, 10,(200, 200), BBoxMode.XY, 1),
        (np.array([400, 400, 400+200, 400+200]), 2, 5,(200, 200), BBoxMode.XY, 1),
        (np.array([200, 200, 200+100, 200+100]), 4, 2.5,(200, 200), BBoxMode.XY, 1),

    ],
)
def test_circle_read_object_mpp(bbox, mpp,mag, expected_center_pixel, slide, bbox_mode, mpp_extract):
    img_mpp = slide.read_object(bbox, mpp=mpp_extract, mpp_bbox = mpp)
    img_mag = slide.read_object(bbox, mpp=mpp_extract, mag_bbox=mag)

    assert img_mpp.size == img_mpp.size
    assert img_mag.mode == img_mag.mode

    diff = ImageChops.difference(img_mpp, img_mag)
    assert diff.getbbox() is None

    pixel = img_mpp.getpixel(expected_center_pixel)[:3]
    assert pixel[0]>=180 # Allow some blurring due to resampling


@pytest.mark.parametrize(
    "bbox, mpp,mag, expected_center_pixel, bbox_mode, mag_extract",
    [

        (BBox(3200, 3200, w=1600, h=1600, mag = 80), 0.25, 40, (200, 200), None, 10),
        (BBox(1600, 1600, w=800, h=800, mag = 40), 0.5, 20, (200, 200), None, 10),

        (BBox(3200, 3200, w=1600, h=1600, mpp = 0.25), 0.25, 40, (200, 200), None, 10),
        (BBox(1600, 1600, w=800, h=800, mpp = 0.5), 0.5, 20, (200, 200), None, 10),
        (BBox(800, 800, w=400, h=400, mpp = 1), 1, 10, (200, 200),  None, 10),
        (BBox(400, 400, w=200, h=200, mpp = 2), 2, 5, (200, 200), None, 10),
        (BBox(200, 200, w=100, h=100, mpp = 4), 4, 2.5, (200, 200), None, 10),

        (BBox(3200, 3200, w=1600, h=1600, mag = 40), 0.25, 40, (200, 200), None, 10),
        (BBox(1600, 1600, w=800, h=800, mag = 20), 0.5, 20, (200, 200), None, 10),
        (BBox(800, 800, w=400, h=400, mag = 10), 1, 10, (200, 200),  None, 10),
        (BBox(400, 400, w=200, h=200, mag = 5), 2, 5, (200, 200), None, 10),
        (BBox(200, 200, w=100, h=100, mag = 2.5), 4, 2.5, (200, 200), None, 10),

        (BBox(3200, 3200, w=1600, h=1600, mag = 40), None, None, (200, 200), None, 10),
        (BBox(1600, 1600, w=800, h=800, mag = 20), None, None, (200, 200), None, 10),
        (BBox(200, 200, w=100, h=100, mpp = 4), None, None, (200, 200),  None, 10),
        (BBox(400, 400, w=200, h=200, mag = 5), None, None, (200, 200), None, 10),
        (BBox(200, 200, w=100, h=100, mag = 2.5), None, None, (200, 200), None, 10),



        (np.array([3200, 3200, 3200+1600, 3200+1600]), 0.25, 40,(200, 200), BBoxMode.XY,10),
        (np.array([1600, 1600, 1600+800, 1600+800]), 0.5, 20,(200, 200), BBoxMode.XY, 10),
        (np.array([800, 800, 800+400, 800+400]), 1, 10,(200, 200), BBoxMode.XY, 10),
        (np.array([400, 400, 400+200, 400+200]), 2, 5,(200, 200), BBoxMode.XY, 10),
        (np.array([200, 200, 200+100, 200+100]), 4, 2.5,(200, 200), BBoxMode.XY, 10),

    ],
)
def test_circle_read_object_mag(bbox, mpp,mag, expected_center_pixel, slide, bbox_mode, mag_extract):
    img_mpp = slide.read_object(bbox, mag=mag_extract, mpp_bbox = mpp)
    img_mag = slide.read_object(bbox, mag=mag_extract, mag_bbox = mag)

    assert img_mpp.size == img_mpp.size
    assert img_mag.mode == img_mag.mode

    diff = ImageChops.difference(img_mpp, img_mag)
    assert diff.getbbox() is None

    pixel = img_mpp.getpixel(expected_center_pixel)[:3]
    assert pixel[0]>=180 # Allow some blurring due to resampling