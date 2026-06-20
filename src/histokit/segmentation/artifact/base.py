from abc import ABC, abstractmethod
from src.histokit.slide.slide import Slide

class ArtifactSegmenter(ABC):

    @abstractmethod
    def segment(self, slide: 'Slide') -> dict:
        ...