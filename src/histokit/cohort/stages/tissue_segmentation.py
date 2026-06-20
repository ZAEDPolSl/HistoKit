from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from .base import BaseCohortPipeline
from ..logger import CohortLogger
from ..builder import build_segmenter
from ...slide.slide import Slide


class TissueDetectionPipeline(BaseCohortPipeline):

    stage_name = "tissue_detection"
    result_subdir = "tissue"

    def output_exists(self, slide_path: Path) -> bool:
        return self.output_path(slide_path).exists()

    def run_one(self, slide_path: Path):
        slide = Slide(slide_path)
        basename = slide_path.stem

        segmenter = build_segmenter(
            algorithm=self.config.algorithm,
            config_path=self.config.config_path,
        )

        return segmenter.segment(
            slide,
            basename=basename,
            verbose=False,
        )