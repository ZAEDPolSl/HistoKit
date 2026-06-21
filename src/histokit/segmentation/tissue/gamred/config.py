from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ...config import BaseSegmentationConfig, CollectorConfig
from ...collectors.image import (
    ThumbnailCollector,
    SegmentationOverlayCollector,
    HistogramCollector,
    ImageOutputCollector,
)
from ...postprocessing.step import (
    Opening,
    FillHoles,
    RemoveSmallRegions,
    PostProcessStep,
)


@dataclass
class GaMRedConfig(BaseSegmentationConfig):
    algorithm_name: ClassVar[str] = "GaMRed"

    collector_registry: ClassVar[dict[str, type]] = {
        "ThumbnailCollector": ThumbnailCollector,
        "SegmentationOverlayCollector": SegmentationOverlayCollector,
        "HistogramCollector": HistogramCollector,
        "ImageOutputCollector": ImageOutputCollector,
    }

    default_collectors: ClassVar[list[CollectorConfig]] = [
        {"name": "ThumbnailCollector"},
        {"name": "SegmentationOverlayCollector"},
        {"name": "HistogramCollector"},
    ]

    vis_mag: float = 1.0
    tissdet_mag: float = 2.5

    thr_min: float = 0.7 * 255
    split_regions: bool = True

    remove_green_pen: bool = True
    thr_green_pen: tuple[int, int] = (15, 120)
    disk_radius_green_pen: int = 9

    remove_black_pen: bool = True
    thr_black_pen: tuple[int, int] = (15, 0)
    disk_radius_black_pen: int = 9

    remove_gray_stains: bool = True

    postprocess_steps: list[PostProcessStep] = field(
        default_factory=lambda: [
            Opening(disk_radius=2),
            FillHoles(enabled=True),
            Opening(disk_radius=2),
            RemoveSmallRegions(thr_area=None),
        ]
    )

    def to_hdf5_dict(self) -> dict[str, Any]:
        data = self.common_hdf5_dict()

        data.update(
            {
                "vis_mag": self.vis_mag,
                "tissdet_mag": self.tissdet_mag,
                "thr_min": self.thr_min,
                "split_regions": self.split_regions,
                "remove_green_pen": self.remove_green_pen,
                "thr_green_pen": self.thr_green_pen,
                "disk_radius_green_pen": self.disk_radius_green_pen,
                "remove_black_pen": self.remove_black_pen,
                "thr_black_pen": self.thr_black_pen,
                "disk_radius_black_pen": self.disk_radius_black_pen,
                "remove_gray_stains": self.remove_gray_stains,
                "postprocess_steps": [
                    step.get_config()
                    for step in self.postprocess_steps
                ],
            }
        )

        return data

    @classmethod
    def _preprocess_dict(cls, data: dict) -> dict:
        if "postprocess_steps" in data:
            data["postprocess_steps"] = cls._build_postprocess_steps(
                data["postprocess_steps"]
            )

        return data

    @staticmethod
    def _build_postprocess_steps(items):
        steps = []

        for item in items:
            if isinstance(item, str):
                name = item
                params = {}

            elif isinstance(item, dict):
                name = item.get("name")
                params = item.get("params", {})

            else:
                continue

            if name == "Opening":
                steps.append(Opening(**params))

            elif name == "FillHoles":
                steps.append(FillHoles(**params))

            elif name == "RemoveSmallRegions":
                steps.append(RemoveSmallRegions(**params))

            else:
                raise ValueError(f"Unknown postprocess step: {name}")

        return steps