from dataclasses import dataclass
from typing import Optional
from src.histo_kit.tissue_segmentation.pipeline.config import PipelineConfig


@dataclass
class GrandQCConfig(PipelineConfig):
    save_confidence_maps: bool = True
    device: str = "cuda:0"
    batch_size: int = 16
    grandqc_path: str = "/mnt/data/Tmp/jmerta/HE/models/GrandQC_MPP1.pth"
    grandqc_mpp: float = 1.0
    patch_size_model: int = 512
    encoder_model: str = 'imagenet'
    encoder_model_weights: str = 'timm-efficientnet-b0'
    overlap: float = 0.75
    blending_mode: str = "gaussian"
    blending_sigma: Optional[float] = None