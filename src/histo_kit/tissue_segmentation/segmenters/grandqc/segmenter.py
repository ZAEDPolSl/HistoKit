from src.histo_kit.tissue_segmentation.segmenters.base import TissueSegmenter
from .config import GrandQCConfig
from ...pipeline.registry import register_segmenter


@register_segmenter("grandqc")
class GrandQC(TissueSegmenter):

    def __init__(self, config: GrandQCConfig):
        self.config = config

    def segment(self, region):
        pass