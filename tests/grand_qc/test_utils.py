import pytest
from PIL import Image
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
from src.grand_qc.utils import make_overlay
Image.MAX_IMAGE_PIXELS = 500_000_000

@pytest.mark.parametrize("region_path, map_tis_path, tis_det_path", [
    ("../../test_data/test_grandqc/region.png", "../../test_data/test_grandqc/map_tis.png", "../../test_data/test_grandqc/tis_det.png")
])
def test_make_overlay(region_path, map_tis_path, tis_det_path):
    region = np.array(Image.open(region_path).convert("RGB"))
    map_tis = Image.open(map_tis_path)
    tis_det = np.array(Image.open(tis_det_path))
    overlay = make_overlay(region, map_tis, tis_det, (int(region.shape[1]*0.1), int(region.shape[0]*0.1)))
    overlay_im = Image.fromarray(overlay)
    overlay_im.save("../../test_data/test_grandqc/overlay.png")


'SS45212_R0A10F2A_190425'