from abc import ABC, abstractmethod
from .collectors.base import PipelineOutput
from ..slide import Slide

class Segmenter(ABC):

    @abstractmethod
    def segment(self, slide: 'Slide') -> dict:
        raise NotImplementedError("Segmenter subclasses must implement the segment method.")


    def _collect(self, name, step, kind, data, **metadata):
        self.output_collector.emit(
            PipelineOutput(
                name=name,
                step=step,
                kind=kind,
                data=data,
                metadata=metadata,
            )
    )