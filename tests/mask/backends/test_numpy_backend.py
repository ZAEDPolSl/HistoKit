import pytest
import numpy as np
from src.histo_kit.mask.backends.numpy import NumpyMaskBackend

@pytest.fixture
def single_mask():
    return np.random.randint(0, 2, (32, 32), dtype=bool)

@pytest.fixture
def mask_list():
    return [
        np.zeros((16, 16), dtype=bool),
        np.ones((16, 16), dtype=np.uint8)
    ]

@pytest.fixture
def properties():
    return {"mag": 20.0, "mpp": 0.25, "note": "test mask"}


def test_init_single_mask(single_mask, properties):
    backend = NumpyMaskBackend(mask=single_mask, properties=properties)
    assert isinstance(backend.mask_array, list)
    assert len(backend.mask_array) == 1
    assert backend.mask_array[0].dtype == np.uint8
    assert backend.mask_dimensions == [(32, 32)]
    assert backend.mask_count == 1
    assert backend.properties == properties
    assert backend.mag == 20.0
    assert backend.mpp == 0.25

def test_init_mask_list(mask_list):
    backend = NumpyMaskBackend(mask=mask_list)
    assert len(backend.mask_array) == 2
    for m in backend.mask_array:
        assert isinstance(m, np.ndarray)
        assert m.ndim == 2
        assert m.dtype == np.uint8
    assert backend.mask_dimensions == [(16, 16), (16, 16)]
    assert backend.mask_count == 2


def test_invalid_single_mask_ndim():
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        NumpyMaskBackend(mask=mask)

def test_invalid_mask_in_list():
    mask_list = [np.zeros((16, 16)), "not a mask"]
    with pytest.raises(TypeError):
        NumpyMaskBackend(mask=mask_list)

def test_invalid_mask_ndim_in_list():
    mask_list = [np.zeros((16, 16)), np.zeros((8, 8, 3))]
    with pytest.raises(ValueError):
        NumpyMaskBackend(mask=mask_list)

def test_invalid_mask_type():
    with pytest.raises(TypeError):
        NumpyMaskBackend(mask="not an array")

def test_dtype_conversion():
    mask = np.random.randint(0, 2, (16, 16), dtype=bool)
    backend = NumpyMaskBackend(mask)
    assert backend.mask_array[0].dtype == np.uint8

    mask_list = [np.zeros((8, 8), dtype=bool), np.ones((8, 8), dtype=np.float32)]
    backend = NumpyMaskBackend(mask_list)
    for m in backend.mask_array:
        assert m.dtype == np.uint8

def test_properties_mag_mpp():
    mask = np.zeros((8, 8))
    props = {"mag": 10.0, "mpp": 0.5}
    backend = NumpyMaskBackend(mask, properties=props)
    assert backend.properties == props
    assert backend.mag == 10.0
    assert backend.mpp == 0.5