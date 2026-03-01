from abc import ABC, abstractmethod
from pathlib import Path
from ...slide.slide import Slide

class VisualizationStep(ABC):

    @abstractmethod
    def __call__(self, data: dict, slide: Slide, out_dir: Path, sub_dir_name: str, vis_mag: float = 0.625):
        pass