from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from openslide import OpenSlide
from src.histo_kit.utils.wsi import slide_info, get_regions_location, load_wsi_mag

ROOT = Path(__file__).parent.parent.parent

def test_get_slide_info():
    svs = f"{ROOT}/test_data/tissue_seg/wsi/TCGA-BK-A6W4-01Z-00-DX1.1C36AB2B-FC3E-4F51-A18A-1B3080E18672.svs"
    width_level_0, height_level_0, obj_power, num_level, vendor, level_downsamples, level_mag, slide_mpp = slide_info(OpenSlide(svs), verbose=True)

    assert width_level_0 == 18088
    assert height_level_0 == 19955
    assert num_level == 3
    assert obj_power == 40
    assert vendor == "aperio"
    assert slide_mpp == 0.2456
    level_mag = [40.0, 9.999248252186332, 4.999624126093166]
    assert_array_equal(level_downsamples, (1.0, 4.000300721732158, 8.000601443464316))
    assert_array_equal(level_mag, [40.0, 9.999248252186332, 4.999624126093166])

def test_get_regions_location():
    mask = np.array([[0, 1, 1, 0],
                     [0, 1, 1, 0],
                     [0, 0, 0, 0]], dtype=bool)
    bbox = get_regions_location(mask)
    gt_bbox = np.array([[0, 1, 2, 3]])
    assert_array_equal(bbox, gt_bbox)

@pytest.mark.parametrize(
    "desired_mag, expected_info",
    [
        (10, "Desired magnification is available"),
        (2.5, "Desired resolution is not available, image will be rescaled from the best level for downsample."),
        (45, "Desired resolution is larger than available, image will be rescaled from the highest magnification available."),
    ]
)
def test_load_wsi_mag(desired_mag, expected_info):
    svs = f"{ROOT}/test_data/tissue_seg/wsi/TCGA-BK-A6W4-01Z-00-DX1.1C36AB2B-FC3E-4F51-A18A-1B3080E18672.svs"

    region, scale_val, info, mpp_slide, ratio = load_wsi_mag(
        OpenSlide(svs),
        desired_mag=desired_mag
    )

    assert region.size == (
        int(18088 * desired_mag / 40),
        int(19955 * desired_mag / 40)
    )
    assert round(scale_val, 1) == round(40 / desired_mag, 1)
    assert mpp_slide == 0.2456
    assert_array_equal(ratio, (1.0, 4.000300721732158, 8.000601443464316))
    assert info == expected_info

def test_read_region():
    pass