from abc import ABC, abstractmethod
from src.histokit.slide.slide import Slide

class TissueSegmenter(ABC):

    @abstractmethod
    def segment(self, slide: 'Slide') -> dict:
        ...