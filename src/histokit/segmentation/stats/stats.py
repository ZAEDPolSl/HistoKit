import numpy as np

from histokit.slide import Slide, BBox, SpatialMask
from collections import Counter

class Stats:
    colors = {
        "Background": [0, 7],
        "Artifact_Free_Tissue": [1],
        "Tissue_Fold": [2],
        "Dark_Spot_Foreign_Object": [3],
        "Pen": [4],
        "Edge_Air_Bubble": [5],
        "Out_of_Focus": [6],
    }

    @staticmethod
    def calculate_stats(art_mask):
        total_counts = Counter()
        region_num = 0

        for mask, bbox in zip(art_mask["mask"], art_mask["bbox"]):
            bbox = BBox(bbox, mag=art_mask["mag_save"])
            mask = SpatialMask(data=mask, bbox=bbox, kind="label")

            classes, counts = np.unique(mask.data, return_counts=True)
            total_counts.update(
                dict(zip(classes.tolist(), counts.tolist()))
            )
            region_num += 1

        width_l0, height_l0 = art_mask["level_dimensions_0"]
        scale = art_mask["mag_save"] / art_mask["mag_l0"]

        width_save = round(width_l0 * scale)
        height_save = round(height_l0 * scale)
        total_pixels = int(width_save * height_save)

        non_background_pixels = sum(
            total_counts[class_id]
            for name, class_ids in Stats.colors.items()
            if name != "Background"
            for class_id in class_ids
        )

        background_pixels = total_pixels - non_background_pixels

        if background_pixels < 0:
            raise ValueError(
                "Non-background pixel count exceeds total slide pixels."
            )

        stats = {
            name: (
                int(background_pixels)
                if name == "Background"
                else int(sum(total_counts.get(class_id, 0) for class_id in class_ids))
            )
            for name, class_ids in Stats.colors.items()
        }

        class_percentages = {
            f"{name}_pct": (
                100.0 * count / non_background_pixels
                if non_background_pixels > 0 else 0.0
            )
            for name, count in stats.items()
            if name != "Background"
        }

        artifact_free_pixels = stats["Artifact_Free_Tissue"]
        artifact_pixels = non_background_pixels - artifact_free_pixels

        artifact_free_pct = (
            100.0 * artifact_free_pixels / non_background_pixels
            if non_background_pixels > 0 else 0.0
        )

        artifact_pct = (
            100.0 * artifact_pixels / non_background_pixels
            if non_background_pixels > 0 else 0.0
        )

        return {
            "region_num": region_num,
            "total_pixels": total_pixels,
            "non_background_pixels": int(non_background_pixels),
            **stats,
            **class_percentages,
            "artifact_pixels": int(artifact_pixels),
            "artifact_pct": artifact_pct,
            'artifact_free_pct': artifact_free_pct,
        }

    @staticmethod
    def get_slide_info(slide_path) -> dict:
        slide = Slide(slide_path)

        width, height = slide.level_dimensions[0]

        return {
            "filepath": str(slide_path),
            "mag_l0": slide.mag,
            "mpp": slide.mpp,
            "width": width,
            "height": height,
            "vendor": slide.properties.get("openslide.vendor"),
        }