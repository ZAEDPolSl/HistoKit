from pathlib import Path
from .base import BaseCohortPipeline
from ..builder import build_segmenter
from ...slide.slide import Slide


class TissueDetectionPipeline(BaseCohortPipeline):
    stage_name = "tissue_detection"
    result_subdir = "tissue_detection"
    result_dir_name = "masks"

    def output_exists(self, slide_path: Path) -> bool:
        return self.output_path(slide_path).exists()

    def run_one(self, slide_path: Path):
        slide = Slide(slide_path, mag=self.config.mag_l0)
        basename = slide_path.stem

        segmenter = build_segmenter(
            algorithm=self.config.algorithm,
            config_path=self.config.config_path,
        )

        segmenter = self.attach_output_collector(segmenter)

        result = segmenter.segment(
            slide,
            basename=basename,
            verbose=False,
            save=False,
        )

        out_path = self.output_path(slide_path)

        self.result_saver().save(
            out_dir=out_path.parent,
            basename=out_path.stem,
            result=result,
        )

        return result