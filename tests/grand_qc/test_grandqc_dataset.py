import os
import pytest
from PIL import Image
import numpy as np
from src.histo_kit.grand_qc.visualisation import make_overlay


@pytest.mark.skip_ci
@pytest.mark.parametrize("region_path, map_tis_path, tis_det_path", [
    ("../../test_data/test_grandqc/region.png", "../../test_data/test_grandqc/map_tis.png", "../../test_data/test_grandqc/tis_det.png")
])
def test_create_grid(region_path, map_tis_path, tis_det_path):
    region = np.array(Image.open(region_path).convert("RGB"))
    map_tis = Image.open(map_tis_path)
    tis_det = np.array(Image.open(tis_det_path))
    overlay = make_overlay(region, map_tis, tis_det, (int(region.shape[1]*0.1), int(region.shape[0]*0.1)))
    overlay_im = Image.fromarray(overlay)
    overlay_im.save("../../test_data/test_grandqc/overlay.png")