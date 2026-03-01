import pytest

from src.histo_kit.utils.wsi import read_object_wsi


@pytest.mark.skip_ci
def test_read_region():
    path = "/mnt/data/Tmp/jmerta/tcga_ucec_svs/TCGA-BG-A0MS-01Z-00-DX1.F496EE7B-68E3-470B-8457-B01D705394C7.svs"
    bbox = [30000, 20000, 2000, 2000]

    mag = 45
    region = read_object_wsi(path, bbox, mag, bbox_mag=20)
    region.save("region_45x.png")

    mag = 20
    region = read_object_wsi(path, bbox, mag, bbox_mag=20)
    region.save("region_20x.png")

    mag = 10
    region = read_object_wsi(path, bbox, mag, bbox_mag=20)
    region.save("region_10x.png")

    mag = 5
    region2 = read_object_wsi(path, bbox, mag, bbox_mag=20)
    region2.save("region_5x.png")