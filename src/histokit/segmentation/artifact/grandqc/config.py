from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ...config import BaseSegmentationConfig, CollectorConfig
from ...collectors.image import (
    ArtifactOverlayCollector,
    ImageOutputCollector,
)

@dataclass
class GrandQCConfig(BaseSegmentationConfig):
    algorithm_name: ClassVar[str] = "GrandQC"

    collector_registry: ClassVar[dict[str, type]] = {
        "ArtifactOverlayCollector": ArtifactOverlayCollector,
        "ImageOutputCollector": ImageOutputCollector,
    }

    default_collectors: ClassVar[list[CollectorConfig]] = [
        {"name": "ArtifactOverlayCollector"},
    ]

    save_confidence_maps: bool = True
    save_raw_mask: bool = True

    device: str = "cuda:0"
    batch_size: int = 16

    model_path: str | Path | None = None

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
            "BG_MODEL": (0, (0, 0, 0)),
        }
    )

    def to_hdf5_dict(self) -> dict[str, Any]:
        data = self.common_hdf5_dict()

        data.update(
            {
                "save_confidence_maps": self.save_confidence_maps,
                "save_raw_mask": self.save_raw_mask,
                "device": self.device,
                "batch_size": self.batch_size,
                "model_path": str(self.model_path)
                if self.model_path is not None
                else None,
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
                "colors": {
                    name: {
                        "id": class_id,
                        "rgb": rgb,
                    }
                    for name, (class_id, rgb) in self.colors.items()
                },
            }
        )

        return data