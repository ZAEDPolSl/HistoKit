from dataclasses import dataclass, field
from ..savers.base import BaseSaver


@dataclass
class PipelineConfig:
    method: str
    vis_mag: float = 0.625
    workers: int = 4
    overwrite: bool = False
    save_mag: float = 5
    out_dir: str = None






