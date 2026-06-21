from PIL import Image
from histokit.slide.mask_utils import merge_regions
from histokit.slide.slide import Slide
from histokit.savers.base import Saver
from histokit.segmentation.artifact.grandqc.config import GrandQCConfig
from histokit.segmentation.artifact.grandqc.segmenter import GrandQCSegmenter
from histokit.segmentation.tissue.gamred.config import GaMRedConfig
from histokit.segmentation.tissue.gamred.segmenter import GaMRedSegmenter

if __name__ == "__main__":
    # Read a WSI in SVS format
    path = r"C:\Repos\HistoKit\Examples\Data\C3N-02274-22.svs"
    s = Slide(path)

    s.assoctiated_images["thumbnail"]


    config = GaMRedConfig()
    res = GaMRedSegmenter(config).segment(s, basename="C3N-02274-22", verbose=True, save = False)

    for key, value in res.items():
        print(f"{key}: {value}")


    s.read_masked_slide(res["bbox"], res["mask"], mag=1, mag_bbox = res["mag_save"])


    config = GrandQCConfig(model_path = "C:\\Repos\\HistoKit\\models\\GrandQC_MPP1.pth")
    #tissue_mask = res,
    res_qc = GrandQCSegmenter(config).segment(slide = s, basename = "C3N-02274-22",  verbose = True, save = False, tissue_mask = res)


    for key, value in res_qc.items():
        print(f"{key}: {value}")


    s.read_masked_slide(res_qc["bbox"], res_qc["mask"], mag=5, mag_bbox = res_qc["mag_save"]).show()