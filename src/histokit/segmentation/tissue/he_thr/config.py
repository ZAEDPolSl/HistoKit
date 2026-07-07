from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Tuple, Union

import numpy as np

from ...config import BaseAlgorithmConfig, CollectorConfig
from ...collectors.image import (
    MaskCollector,
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
class HeThrConfig(BaseAlgorithmConfig):
    algorithm_name: ClassVar[str] = "HeThr"

    collector_registry: ClassVar[dict[str, type]] = {
        "ThumbnailCollector": ThumbnailCollector,
        "SegmentationOverlayCollector": SegmentationOverlayCollector,
        "HistogramCollector": HistogramCollector,
        "ImageOutputCollector": ImageOutputCollector,
        "MaskCollector": MaskCollector,
    }

    default_collectors: ClassVar[list[CollectorConfig]] = [
        {"name": "ThumbnailCollector"},
        {"name": "SegmentationOverlayCollector"},
        {"name": "MaskCollector"},
    ]

    vis_mag: float = 1.0
    tissdet_mag: float = 2.5
    blur_kernel_width: int = 0
    nbins: int = 256
    
    postprocess_steps: list[PostProcessStep] = field(
        default_factory=lambda: [
        ]
    )

    def to_hdf5_dict(self) -> dict[str, Any]:
        data = self.common_hdf5_dict()

        data.update(
            {
                "vis_mag": self.vis_mag,
                "tissdet_mag": self.tissdet_mag,
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