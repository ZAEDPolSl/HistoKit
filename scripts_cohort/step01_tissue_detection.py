
from histokit.savers.base import Saver
from histokit.segmentation.artifact.grandqc.config import GrandQCConfig
from histokit.segmentation.artifact.grandqc.segmenter import GrandQCSegmenter
from histokit.segmentation.tissue.gamred.config import GaMRedConfig
from histokit.segmentation.tissue.gamred.segmenter import GaMRedSegmenter
from histokit.slide import Slide


if __name__ == "__main__":
    slide = "tests\data\wsi\Aperio\CMU-1.svs"

    slide = Slide(slide)

    yaml = "scripts_cohort\config\gamred.yaml"
    config = GaMRedConfig.from_yaml(yaml)

    #res = GaMRedSegmenter(config).segment(slide, basename="CMU-1", verbose=True)

    res = Saver("hdf5").load("C:\Repos\HistoKit\outputs\mask_gamred\CMU-1.h5")
    config = GrandQCConfig.from_yaml("scripts_cohort\config\grandqc.yaml")
    res = GrandQCSegmenter(config).segment(slide, basename="CMU-1", verbose=True)
