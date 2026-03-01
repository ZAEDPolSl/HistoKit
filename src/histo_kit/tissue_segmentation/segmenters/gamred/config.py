from dataclasses import dataclass, field
from typing import List, Callable
from ...postprocessing.steps import Opening, FillHoles, RemoveSmallObjects, PostProcessStep
from ...visualisation.overlay import Thumbnail, TissueSegmentation
from ...visualisation.plots import ThresholdingHistograms
from ....tissue_segmentation.pipeline.config import PipelineConfig

def default_postprocessing():
    return [
        Opening(),
        FillHoles(),
        Opening(),
        RemoveSmallObjects(),
    ]

def default_visualisation():
    return [
        ThresholdingHistograms(),
        Thumbnail(),
        TissueSegmentation(),

    ]

@dataclass
class GaMRedConfig(PipelineConfig):
    tissdet_mag: float = 2.5
    fill_holes: bool = True
    open_disk_radius: int = 2
    close_disk_radius: int = 2
    remove_small_obj: bool = True
    thr_min: float = 0.7 * 255
    postprocess_steps: List[PostProcessStep] = field(default_factory=default_postprocessing)







