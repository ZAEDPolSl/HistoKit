from pathlib import Path

import numpy as np
import pytest

from src.histo_kit.slide.slide import Slide
from src.histo_kit.tissue_segmentation.segmenters.gamred.config import GaMRedConfig
from src.histo_kit.tissue_segmentation.segmenters.gamred.segmenter import GaMRedSegmenter


TEST_PATH = Path(__file__).parent.parent.parent / "data"

@pytest.fixture(params=["sample_aperio_cptac_ucec.svs", "sample_aperio_tcga_ucec.svs", "sample_hammamatsu_rosella.ndpi", "sample_layered_endo.tif"])
def slide(tmp_path: Path, request):
    path = TEST_PATH / f"{request.param}"
    return Slide(str(path))

def test_gamred_segmenter(slide):

    config = GaMRedConfig(tissdet_mag=2.5, method="gamred")
    segmenter = GaMRedSegmenter(config)
    result = segmenter.segment(slide)
    print(slide.level_dimensions)
    print(slide.level_mag)
    print(result.size)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)
    assert np.all(result == 1)