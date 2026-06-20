from pathlib import Path
from .base import BaseCohortPipeline
from ..builder import build_segmenter
from ...slide.slide import Slide
from ...savers.base import Saver


class ArtifactDetectionPipeline(BaseCohortPipeline):
    stage_name = "artifact_detection"
    result_subdir = "artifact"

    def output_exists(self, slide_path: Path) -> bool:
        return self.output_path(slide_path).exists()

    def get_tissue_source_algorithm(self) -> str | None:
        tissue_source = getattr(self.config, "tissue_source", None)

        if tissue_source is None:
            return None

        if isinstance(tissue_source, dict):
            return tissue_source.get("algorithm")

        return getattr(tissue_source, "algorithm", None)

    def tissue_mask_path(self, slide_path: Path):

        tissue_algorithm = self.get_tissue_source_algorithm()

        if tissue_algorithm is None:
            return None

        return self.output_path(
            slide_path,
            algorithm=tissue_algorithm,
        )

    def load_tissue_mask(self, slide_path: Path):
        path = self.tissue_mask_path(slide_path)

        if path is None or not path.exists():
            return None

        return self.result_saver().load(path)

    def run_one(self, slide_path: Path):
        slide = Slide(slide_path)
        basename = slide_path.stem

        tissue_mask = self.load_tissue_mask(slide_path)

        segmenter = build_segmenter(
            algorithm=self.config.algorithm,
            config_path=self.config.config_path,
        )

        return segmenter.segment(
            slide,
            basename=basename,
            tissue_mask=tissue_mask,
            verbose=False,
        )