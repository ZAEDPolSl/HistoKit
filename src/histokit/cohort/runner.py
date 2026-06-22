from __future__ import annotations

from .config import CohortConfig
from .pipelines.tissue import TissueDetectionPipeline
from .pipelines.artifact import ArtifactDetectionPipeline


class CohortRunner:
    def __init__(self, config: CohortConfig):
        self.config = config

    def run(self):
        if self.config.tissue_detection is not None:
            print("=> Running tissue detection pipeline...")
            print(f"Input directory: {self.config.input_dir}")
            print(f"Output directory: {self.config.output_dir}")
            tissue_pipeline = TissueDetectionPipeline(
                self.config.tissue_detection
            )
            tissue_pipeline.run()

        if self.config.artifact_detection is not None:
            print("=> Running artifact detection pipeline...")
            print(f"Input directory: {self.config.input_dir}")
            print(f"Output directory: {self.config.output_dir}")
            artifact_pipeline = ArtifactDetectionPipeline(
                self.config.artifact_detection
            )
            artifact_pipeline.run()

        if self.config.statistics is not None:
            print("=> Calculating statistics...")
            print(f"Input directory: {self.config.input_dir}")
            print(f"Output directory: {self.config.output_dir}")
            raise NotImplementedError(
                "Statistics pipeline is not implemented yet."
            )