from __future__ import annotations

from .config import CohortConfig
from .pipelines.tissue import TissueDetectionPipeline
from .pipelines.artifact import ArtifactDetectionPipeline


class CohortRunner:
    def __init__(self, config: CohortConfig):
        self.config = config

    def run(self):
        if self.config.tissue_detection is not None:
            tissue_pipeline = TissueDetectionPipeline(
                self.config.tissue_detection
            )
            tissue_pipeline.run()

        if self.config.artifact_detection is not None:
            artifact_pipeline = ArtifactDetectionPipeline(
                self.config.artifact_detection
            )
            artifact_pipeline.run()

        if self.config.statistics is not None:
            raise NotImplementedError(
                "Statistics pipeline is not implemented yet."
            )