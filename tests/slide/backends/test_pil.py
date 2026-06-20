import pytest
from pathlib import Path
from PIL import Image
from histokit.slide.backends.pil import PILBackend

TEST_PATH = Path(__file__).parent.parent.parent / "data"

@pytest.fixture(params=["sample_tnbc.png", "sample_ocelot.jpg", "sample_single_rosella.tiff"])
def backend(tmp_path: Path, request):
    path = TEST_PATH / f"{request.param}"
    return PILBackend(str(path))

def test_backend_initializes(backend):
    assert backend.level_count == 1
    assert len(backend.level_downsamples) == backend.level_count

def test_level_dimensions_valid(backend):
    dims = backend.level_dimensions
    assert isinstance(dims, list)
    assert isinstance(dims[0], tuple)
    assert len(dims[0]) == 2

def test_read_region_returns_image(backend):
    dims = backend.level_dimensions[0]
    w, h = dims

    region = backend.read_region(
        location=(0, 0),
        level=0,
        size=(min(512, w), min(512, h))
    )

    assert isinstance(region, Image.Image)
    assert region.size[0] > 0
    assert region.size[1] > 0

def test_mag_mpp_are_valid_or_none(backend):
    if backend.mag is not None:
        assert isinstance(backend.mag, float)
        assert backend.mag > 0

    if backend.mpp is not None:
        assert isinstance(backend.mpp, float)
        assert backend.mpp > 0

def test_best_level_for_downsample(backend):
    level = backend.get_best_level_for_downsample(4.0)

    assert isinstance(level, int)
    assert 0 <= level < backend.level_count

def test_region_mode(backend):
    region = backend.read_region((0, 0), 0, (256, 256))
    assert region.mode in ["RGBA", "RGB"]

def test_level_downsamples(backend):
    assert backend.level_downsamples == [1.0]
