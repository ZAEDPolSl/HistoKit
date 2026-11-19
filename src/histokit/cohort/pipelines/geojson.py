from pathlib import Path
from .base import BaseCohortPipeline
from ...savers.hdf52geojson import masks_to_geojson, masks_to_patch_coords


class GeojsonPipeline(BaseCohortPipeline):
    stage_name = "geojson"
    result_subdir = "geojson"
    result_dir_name = "geojson"

    def output_exists(self, slide_path: Path) -> bool:
        return self.output_path(slide_path).exists()

    def set_art_src_from_pipeline(self, art_src):
        self.art_src = art_src

    def get_artifact_results(self) -> str | None:
        artifact_dir = getattr(self.config, "artifact_source", None)
        print(f"Artifact dir: {artifact_dir}")

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
        output_path = str(self.output_path(slide_path)).replace("h5", "geojson")

        geojson = masks_to_geojson(
            art_mask=art_mask,
            output_path=output_path,
            remove_holes=False,
            include_classes=[1, 2, 3, 4, 5, 6],
            min_area=10000,
            min_hole=10000
        )

        output_path = str(self.output_path(slide_path)).replace(".h5", "_patches.h5")
        output_preview = str(self.output_path(slide_path)).replace("h5", "png")
        coords = masks_to_patch_coords(
            art_mask=art_mask,
            output_h5=output_path,
            output_preview=output_preview,
            patch_size=512,
            target_mag=10,
            include_classes=(1, 2, 3, 4, 5, 6),
            min_tissue_fraction=0.95,
            overlap=0,
        )

        return geojson