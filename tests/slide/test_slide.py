import numpy as np
import pytest
from pathlib import Path
from histokit.savers import Saver
from histokit.slide import NumpyBackend, OpenSlideBackend, PILBackend, Slide

TEST_PATH = Path(__file__).parent.parent / "data/wsi/"
MASKS_PATH = Path(__file__).parent.parent / "data/tissue_masks/"

@pytest.fixture(params=[
    lambda: np.random.choice([False, True], (64, 64)).astype(bool),          # Binary bool
    lambda: np.random.randint(0, 2, (64, 64), dtype=np.uint8),  # Binary 0-1
    lambda: np.random.choice([0, 255], (64, 64)).astype(np.uint8), # Binary 0-255
    lambda: np.random.randint(0, 256, (64, 64), dtype=np.uint8),  # Grayscale uint8
    lambda: np.random.rand(64, 64).astype(np.float32),           # Grayscale float 0-1
    lambda: np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8), # RGB
    lambda: np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8), # RGBA
])
def slide_numpy(request):
    return Slide(request.param())

@pytest.fixture(params=["sample_tnbc.png", "sample_ocelot.jpg", "sample_single_rosella.tiff"])
def slide_pil(tmp_path: Path, request):
    path = TEST_PATH / f"{request.param}"
    return Slide(str(path))

@pytest.fixture(params=["Aperio/CMU-1-Small-Region.svs", "ARGOS/Argos-1-Stacked.avs", "Philips/Philips-1.tiff", "Hamamatsu/Hamamatsu-1.ndpi"])
def slide_openslide(tmp_path: Path, request):
    path = TEST_PATH / f"{request.param}"
    return Slide(str(path))

@pytest.fixture(params=[
    "sample_tnbc.png",
    "sample_ocelot.jpg",
    "Aperio/CMU-1-Small-Region.svs",
    "ARGOS/Argos-1-Stacked.avs",
    "Philips/Philips-1.tiff",
    "Hamamatsu/Hamamatsu-1.ndpi",
    lambda: np.random.choice([False, True], (64, 64)).astype(bool)
])
def slide(tmp_path: Path, request):
    param = request.param

    if callable(param):
        data = param()
    else:
        data = str(TEST_PATH / param)

    return Slide(data)

@pytest.fixture(params=["Aperio/CMU-1-Small-Region.svs","ARGOS/Argos-1-Stacked.avs", "Philips/Philips-1.tiff", "Hamamatsu/Hamamatsu-1.ndpi", lambda: np.random.choice([False, True], (64, 64)).astype(bool)])
def slide_test_extraction(tmp_path: Path, request):
    param = request.param
    if callable(param):
        data = param()
    else:
        data = str(TEST_PATH / param)

    return Slide(data)

def test_backend_initializes_numpy(slide_numpy):
    assert isinstance(slide_numpy.backend, NumpyBackend)
    assert slide_numpy.level_count == 1
    assert slide_numpy.level_downsamples == [1.0]

def test_backend_initializes_pil(slide_pil):
    assert isinstance(slide_pil.backend, PILBackend)
    assert slide_pil.level_count == 1
    assert slide_pil.level_downsamples == [1.0]

def test_backend_initializes_openslide(slide_openslide):
    assert isinstance(slide_openslide.backend, OpenSlideBackend)
    assert slide_openslide.level_count >= 1
    assert len(slide_openslide.level_downsamples) == slide_openslide.level_count

def test_level_dimensions_valid(slide):
    dims = slide.level_dimensions
    assert isinstance(dims, list)
    assert isinstance(dims[0], tuple)
    assert len(dims[0]) == 2

def test_mag_mpp_are_valid_or_none(slide):
    if slide.mag is not None:
        assert isinstance(slide.mag, float)
        assert slide.mag > 0

    if slide.mpp is not None:
        assert isinstance(slide.mpp, float)
        assert slide.mpp > 0

def test_get_properties(slide):
    props = slide.properties
    assert isinstance(props, dict)

@pytest.mark.parametrize("ratio, expected, slide", [(2.5, 0, "Aperio/CMU-1-Small-Region.svs"), (2.5, 1, "ARGOS/Argos-1-Stacked.avs")])
def test_get_best_level_for_downsample_ratio(ratio, expected, slide):
    slide = Slide(str(TEST_PATH / slide))
    level = slide.get_best_level_for_downsample(ratio=ratio)
    assert isinstance(level, int)
    assert expected == level

@pytest.mark.parametrize("mpp, expected, slide", [(0.4597, 0, "Aperio/CMU-1-Small-Region.svs"), (1.9, 2, "ARGOS/Argos-1-Stacked.avs")])
def test_get_best_level_for_downsample_mpp(mpp, expected, slide):
    slide = Slide(str(TEST_PATH / slide))
    level = slide.get_best_level_for_downsample(mpp=mpp)
    assert isinstance(level, int)
    assert expected == level

@pytest.mark.parametrize("mag, expected, slide", [(2.5, 0, "Aperio/CMU-1-Small-Region.svs"), (4, 2, "ARGOS/Argos-1-Stacked.avs")])
def test_get_best_level_for_downsample_mag(mag, expected, slide):
    slide = Slide(str(TEST_PATH / slide))
    level = slide.get_best_level_for_downsample(mag=mag)
    assert isinstance(level, int)
    assert expected == level





