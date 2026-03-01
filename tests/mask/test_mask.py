import pytest
import numpy as np
from PIL import Image

from src.histo_kit.mask.mask import Mask
from src.histo_kit.slide.bbox import BBox
from tests.mocks.mock_slide import MockSlide
from tests.mocks.mock_wsi_backend import MockWSIBackend


@pytest.fixture
def simple_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 1
    return mask

def real_mask():
    img = Image.new("L", (100, 100), color=0)
    for x in range(30, 70):
        for y in range(30, 70):
            img.putpixel((x, y), 1)
    return np.array(img)

def real_slide():
    backend = MockWSIBackend()
    return MockSlide(backend)


@pytest.fixture
def mock_slide():
    backend = MockWSIBackend()
    return MockSlide(backend)

def test_mask_initialization(simple_mask):
    mask = Mask(simple_mask)

    assert mask.mask_count == 1
    assert mask.mask_dimensions == [(100, 100)]
    assert mask.exclude_values == [0]


def test_mask_values(simple_mask):
    mask = Mask(simple_mask)
    assert mask.mask_values == [0, 1]

def test_exclude_values_bool():
    mask_array = np.ones((10, 10), dtype=np.uint8)
    mask = Mask(mask_array, exclude_values=[True, False])
    assert mask.exclude_values == [1, 0]


def test_exclude_values_invalid():
    mask_array = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(TypeError):
        Mask(mask_array, exclude_values="invalid")

def test_scale_resolution_from_mag():
    mask_array = np.ones((10, 10), dtype=np.uint8)
    mask = Mask(mask_array, mag=20)
    assert mask.mag == 20
    assert mask.mpp == pytest.approx(mask.REF_MAG * mask.REF_MPP / 20)


def test_scale_resolution_from_mpp():
    mask_array = np.ones((10, 10), dtype=np.uint8)
    mask = Mask(mask_array, mpp=0.5)
    assert mask.mpp == 0.5
    assert mask.mag == pytest.approx(mask.REF_MAG * mask.REF_MPP / 0.5)

def test_mask_region_applies_mask(simple_mask, mock_slide):
    bbox = BBox(1600, 1600, w=800, h=800)
    mask = Mask(simple_mask, bbox=bbox)

    result = mask.mask_region(mock_slide)

    assert isinstance(result, np.ndarray)
    assert result.shape[2] == 4  # RGBA
    assert result.dtype == np.uint8

    assert np.any(result == 0)


def test_mask_region_applies_mask_real_data(real_mask, real_slide):
    bbox = BBox(1600, 1600, w=800, h=800)
    mask = Mask(simple_mask, bbox=bbox)
    result = mask.mask_region(mock_slide)
    assert isinstance(result, np.ndarray)
    assert result.shape[2] == 4  # RGBA
    assert result.dtype == np.uint8
    assert np.any(result == 0)

def test_invalid_backend_type():
    with pytest.raises(TypeError):
        Mask("unsupported_type")