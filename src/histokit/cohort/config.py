from dataclasses import dataclass


@dataclass
class StageConfig:
    algorithm: str
    config_path: str
    workers: int = 1
    device: str = "cpu"


@dataclass
class CohortConfig:

    input_dir: str
    output_dir: str

    tissue_detection: StageConfig | None = None

    artifact_detection: StageConfig | None = None

    statistics: StageConfig | None = None





