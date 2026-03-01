from dataclasses import dataclass, field
from typing import List
from ..visualisation.base import VisualizationStep
from ...tissue_segmentation.savers.base_server import BaseSaver

@dataclass
class PipelineConfig:
    method: str
    vis_mag: float = 0.625
    saver: BaseSaver = None,
    workers: int = 4
    overwrite: bool = False
    save_mag: float = 1
    out_dir: str = None
    visualization_steps: List[VisualizationStep] = field(default_factory=list)



