import pytest
import numpy as np
from PIL import Image
from src.histo_kit.slide.backends.numpy import NumpyBackend

@pytest.fixture(params=[
    lambda: np.random.choice([False, True], (64, 64)).astype(bool),          # Binary bool
    lambda: np.random.randint(0, 2, (64, 64), dtype=np.uint8),  # Binary 0-1
    lambda: np.random.choice([0, 255], (64, 64)).astype(np.uint8), # Binary 0-255
    lambda: np.random.randint(0, 256, (64, 64), dtype=np.uint8),  # Grayscale uint8
    lambda: np.random.rand(64, 64).astype(np.float32),           # Grayscale float 0-1
    lambda: np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8), # RGB
    lambda: np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8), # RGBA
])
def backend(request):
    arr = request.param()
    return NumpyBackend(arr)

def test_backend_initializes(backend):
    assert backend.level_count == 1
    assert len(backend.level_downsamples) == backend.level_count
    assert "mode" in backend.properties
    assert isinstance(backend.properties["mode"], str)

def test_level_dimensions_valid(backend):
    dims = backend.level_dimensions
    assert isinstance(dims, list)
    w, h = dims[0]
    assert isinstance(w, int) and w > 0
    assert isinstance(h, int) and h > 0

def test_read_region_returns_image(backend):
    w, h = backend.level_dimensions[0]
    region = backend.read_region(
        location=(0, 0),
        level=0,
        size=(min(32, w), min(32, h))
    )
    assert isinstance(region, Image.Image)
    assert region.size[0] > 0
    assert region.size[1] > 0

def test_mag_mpp_are_none(backend):
    assert backend.mag is None
    assert backend.mpp is None

def test_region_mode_matches_expected(backend):
    region = backend.read_region((0, 0), 0, (32, 32))
    expected_mode = backend.properties["mode"]
    assert region.mode == expected_mode

def test_level_downsamples(backend):
    assert backend.level_downsamples == [1.0]

def test_read_region_full_dimensions(backend):
    w, h = backend.level_dimensions[0]
    region = backend.read_region((0, 0), 0, (w, h))
    assert region.size == (w, h)