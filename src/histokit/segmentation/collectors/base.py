from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import numpy as np
from enum import Enum
from typing import Any

class OutputKind(str, Enum):
    IMAGE = "image"
    MASK = "mask"
    HISTOGRAM = "histogram"
    METADATA = "metadata"

@dataclass
class PipelineOutput:
    name: str
    step: str
    data: Any
    kind: OutputKind
    metadata: dict = field(default_factory=dict)

class OutputCollector:
    def emit(self, output: PipelineOutput) -> None:
        raise NotImplementedError

class NoOpOutputCollector:
    def emit(self, output: PipelineOutput) -> None:
        pass

class CompositeOutputCollector(OutputCollector):
    def __init__(self, collectors: list[OutputCollector]):
        self.collectors = collectors

    def emit(self, output: PipelineOutput) -> None:
        for collector in self.collectors:
            collector.emit(output)

