import os
import pytest
import numpy as np
#from src.histo_kit.grand_qc.statistics import calculate_stats
import scipy.io as sio
from matplotlib import pyplot as plt

from src.histo_kit.grand_qc.artifacts import Artifact
from src.histo_kit.grand_qc.statistics import calculate_stats
from src.histo_kit.grand_qc.visualisation import make_artifacts_color_map


@pytest.mark.skipif(os.getenv("CI")=="true", reason="Large tissue files not uploaded to CI")
@pytest.mark.parametrize("mask_grandqc_f, conf_map_f", [
    ("../../test_data/test_grandqc/11-14_he_grandqc_mask.mat", "../../test_data/test_grandqc/11-14_he_confidence_map.mat")
])
def test_calculate_stats(mask_grandqc_f, conf_map_f):

    mask_mat = sio.loadmat(mask_grandqc_f)
    df = calculate_stats(mask_mat)
    df.to_csv("test.csv", index=False)

