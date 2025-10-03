import pytest

from src.wsi_utils.heatmaps import patch_wsi, rescale_wsi


@pytest.mark.parametrize("wsi_path, patch_size, overlap, mirroring_type, mag, out_folder", [
("../../test_data/test_utils/region_1.tif",256, 90, "skip", 10, "out"),
])
def test_patch_image(wsi_path, patch_size, overlap, mirroring_type, mag, out_folder):
    patches = patch_wsi(wsi_path, patch_size, overlap, mirroring_type, mag, out_folder)

@pytest.mark.parametrize("wsi, desired_mag, rescale_method, verbose, allow_upscaling", [
("../../test_data/test_utils/region_1.tif",256, 90, "skip", 10, "out"),
])
def test_rescale_wsi(wsi, desired_mag, rescale_method, verbose, allow_upscaling):
    patches, scale_val, info  = rescale_wsi(wsi, desired_mag, rescale_method, verbose, allow_upscaling)
