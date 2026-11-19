import os
from pathlib import Path
from .base import BaseCohortPipeline
from ...segmentation.stats.stats import Stats
from ...slide.slide import Slide
import pandas as pd

class StatsPipeline(BaseCohortPipeline):
    stage_name = "statistics"
    result_subdir = "statistics"
    result_dir_name = "statistics"

    def output_exists(self, slide_path: Path) -> bool:
        return self.output_path(slide_path).exists()

    def set_art_src_from_pipeline(self, art_src):
        self.art_src = art_src

    def get_artifact_results(self) -> str | None:
        artifact_dir = getattr(self.config, "artifact_source", None)

        if artifact_dir is None:
            return self.art_src

        if isinstance(artifact_dir, dict):
            return artifact_dir.get("dir")

        return getattr(artifact_dir, "dir", None)

    def art_mask_path(self, slide_path: Path) -> Path | None:
        mask_dir = self.get_artifact_results()

        if mask_dir is None:
            return None

        return self.result_path(
            mask_dir=mask_dir,
            slide_path=slide_path
        )

    def load_art_mask(self, slide_path: Path) -> dict | None:
        path = self.art_mask_path(slide_path)

        if path is None or not path.exists():
            return None

        return self.result_saver().load(path)

    def run_one(self, slide_path: Path) -> dict:
        print(f"Loading slide: {slide_path}")

        art_mask = self.load_art_mask(slide_path)
        slide_info = Stats.get_slide_info(slide_path)
        stats = Stats.calculate_stats(art_mask)

        result = {
            "basename": Path(slide_path).stem,
            **slide_info,
            **stats,
        }

        csv_path = Path(self.config.output_dir) / "slide_stats.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([result])

        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        df.to_csv(
            csv_path,
            mode="a",
            header=not file_exists,
            index=False,
        )

        return result