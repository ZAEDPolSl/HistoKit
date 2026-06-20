from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Any
import yaml

from ...collectors.base import CompositeOutputCollector
from ...collectors.image import ThumbnailCollector, SegmentationOverlayCollector, HistogramCollector, ImageOutputCollector
from ...postprocessing.step import Opening, FillHoles, RemoveSmallRegions, PostProcessStep

COLLECTOR_REGISTRY = {
    "ThumbnailCollector": ThumbnailCollector,
    "SegmentationOverlayCollector": SegmentationOverlayCollector,
    "HistogramCollector": HistogramCollector,
    "ImageOutputCollector": ImageOutputCollector,
}



@dataclass
class GaMRedConfig:

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

    fill_holes: bool = True
    open_disk_radius: int = 2
    close_disk_radius: int = 2
    remove_small_regions: bool = True
    small_regions_thr: int | None = None

    saver: str = "hdf5"
    out_dir: str | Path | None = None

    postprocess_steps: list[PostProcessStep] = field(default_factory=list)

    collectors: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.postprocess_steps:
            self.postprocess_steps = [
                Opening(disk_radius=self.open_disk_radius),
                FillHoles(enabled=self.fill_holes),
                Opening(disk_radius=self.open_disk_radius),
                RemoveSmallRegions(thr_area=self.small_regions_thr),
            ]

        if not self.collectors:
            self.collectors = [
                {"name": "ThumbnailCollector"},
                {"name": "SegmentationOverlayCollector"},
                {"name": "HistogramCollector"},
            ]

    def build_output_collector(self):
        if self.out_dir is None:
            raise ValueError("out_dir must be provided to build collectors.")

        collector_instances = []

        for item in self.collectors:
            if isinstance(item, str):
                name = item
                params = {}
            elif isinstance(item, dict):
                name = item.get("name")
                params = item.get("params", {})
            else:
                raise TypeError(f"Invalid collector config: {item}")

            try:
                collector_cls = COLLECTOR_REGISTRY[name]
            except KeyError:
                raise ValueError(f"Unknown output collector: {name}")

            collector_instances.append(
                collector_cls(
                    out_dir=self.out_dir,
                    **params,
                )
            )

        return CompositeOutputCollector(collector_instances)

    def to_hdf5_dict(self) -> dict[str, Any]:
        return {
            "tissdet_mag": self.tissdet_mag,
            "thr_min": self.thr_min,
            "remove_green_pen": self.remove_green_pen,
            "thr_green_pen": self.thr_green_pen,
            "disk_radius_green_pen": self.disk_radius_green_pen,
            "remove_black_pen": self.remove_black_pen,
            "thr_black_pen": self.thr_black_pen,
            "disk_radius_black_pen": self.disk_radius_black_pen,
            "remove_gray_stains": self.remove_gray_stains,
            "fill_holes": self.fill_holes,
            "open_disk_radius": self.open_disk_radius,
            "remove_small_regions": self.remove_small_regions,
            "small_regions_thr": self.small_regions_thr,
            "postprocess_steps": [
                step.get_config()
                for step in self.postprocess_steps
            ],
            "collectors": self.collectors,
        }

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            "name": "GaMRed",
            "config": self.to_hdf5_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GaMRedConfig":
        data = data or {}

        if "postprocess_steps" in data:
            data["postprocess_steps"] = cls._build_postprocess_steps(
                data["postprocess_steps"]
            )

        field_names = {f.name for f in fields(cls)}
        filtered = {
            k: v
            for k, v in data.items()
            if k in field_names
        }

        return cls(**filtered)

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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GaMRedConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return cls.from_dict(data)