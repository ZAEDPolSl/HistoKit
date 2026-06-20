from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
import yaml

from ...collectors.base import CompositeOutputCollector
from ...collectors.image import ArtifactOverlayCollector, ImageOutputCollector


COLLECTOR_REGISTRY = {
    "ArtifactOverlayCollector": ArtifactOverlayCollector,
    "ImageOutputCollector": ImageOutputCollector,
}


@dataclass
class GrandQCConfig:

    saver: str | None = "hdf5"
    out_dir: str | Path | None = "./outputs"
    save_mag: float | None = 1.0

    save_confidence_maps: bool = True
    save_raw_mask: bool = True

    device: str = "cuda:0"
    batch_size: int = 16

    model_path: str | Path = "models/GrandQC_MPP1.pth"

    grandqc_mpp: float = 1.0
    det_mag: float = 10.0
    vis_mag: float = 1.0

    patch_size_model: int = 512
    patch_size: int = 512

    encoder_weights: str = "imagenet"
    encoder: str = "timm-efficientnet-b0"

    overlap: float = 0.75
    blending_mode: str = "gaussian"
    blending_sigma: float | None = None

    num_workers: int = 4
    pad_value: int = 255
    grid_offset: float = 0.5
    classes: int = 8

    colors: dict[str, tuple[int, tuple[int, int, int]]] = field(
        default_factory=lambda: {
            "BG_THR": (0, (0, 0, 0)),
            "NORM": (1, (128, 128, 128)),
            "ART_FOLD": (2, (255, 99, 71)),
            "ART_DARKSPOT": (3, (0, 255, 0)),
            "ART_PEN": (4, (255, 0, 0)),
            "ART_EDGE": (5, (255, 0, 255)),
            "ART_FOCUS": (6, (75, 0, 130)),
            "BG_MODEL": (7, (50, 120, 230)),
        }
    )

    collectors: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.collectors:
            self.collectors = [
                {"name": "ArtifactOverlayCollector"},
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
            "saver": self.saver,
            "out_dir": str(self.out_dir) if self.out_dir is not None else None,
            "save_confidence_maps": self.save_confidence_maps,
            "save_raw_mask": self.save_raw_mask,
            "device": self.device,
            "batch_size": self.batch_size,
            "model_path": str(self.model_path),
            "grandqc_mpp": self.grandqc_mpp,
            "det_mag": self.det_mag,
            "vis_mag": self.vis_mag,
            "patch_size_model": self.patch_size_model,
            "patch_size": self.patch_size,
            "encoder_weights": self.encoder_weights,
            "encoder": self.encoder,
            "overlap": self.overlap,
            "blending_mode": self.blending_mode,
            "blending_sigma": self.blending_sigma,
            "num_workers": self.num_workers,
            "pad_value": self.pad_value,
            "grid_offset": self.grid_offset,
            "classes": self.classes,
            "colors": {name: {"id": class_id, "rgb": rgb}
                for name, (class_id, rgb) in self.colors.items()
            },
            "collectors": self.collectors,
        }

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            "name": "GrandQC",
            "config": self.to_hdf5_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GrandQCConfig":
        data = data or {}

        field_names = {f.name for f in fields(cls)}

        filtered = {
            k: v
            for k, v in data.items()
            if k in field_names
        }

        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GrandQCConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return cls.from_dict(data)