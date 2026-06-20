from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from ....cohort_runner.config import PipelineConfig


@dataclass
class GrandQCConfig(PipelineConfig):

    saver:str = "hdf5"
    save_confidence_maps: bool = True
    device: str = "cuda:0"
    batch_size: int = 16
    model_path: str = "/mnt/data/Tmp/jmerta/HE/models/GrandQC_MPP1.pth"
    grandqc_mpp: float = 1.0
    det_mag: float = 10.0
    patch_size_model: int = 512
    encoder_weights: str = 'imagenet'
    encoder: str = 'timm-efficientnet-b0'
    overlap: float = 0.75
    blending_mode: str = "gaussian"
    blending_sigma: Optional[float] = None
    num_workers: int = 4
    vis_mag: float = 1.0
    patch_size: int = 512
    pad_value: int = 255
    grid_offset: float = 0.5
    classes: int = 8
    save_raw_mask: int = True

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

    visualisation_steps: list[Callable] = (
        ArtifactOverlay(),
    )

    def to_hdf5_dict(self) -> dict[str, Any]:
        return {
            "det_mag": self.det_mag,
        }

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            "name": "GrandQC",
            "config": self.to_hdf5_dict(),
        }


