from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SlidesConfig:
    input_dir: str
    file_list: str | list[str] | None = None
    pattern: str = "*.svs"


@dataclass
class LoggingConfig:
    enabled: bool = True
    log_dir: str | None = None
    processed_file: str = "processed.csv"
    errors_file: str = "errors.csv"


@dataclass
class SourceConfig:
    algorithm: str


@dataclass
class StageConfig:
    enabled: bool = True

    algorithm: str | None = None
    config_path: str | None = None

    parallel_workers: int = 1
    overwrite: bool = False

    tissue_source: SourceConfig | None = None
    artifact_source: SourceConfig | None = None

    input_dir: str = "./data/slides"
    file_list: str | list[str] | None = None
    pattern: str = "*.svs"

    output_dir: str = "./outputs"
    saver: str | None = "hdf5"

    logging: LoggingConfig | None = None


@dataclass
class CohortConfig:
    slides: SlidesConfig
    output_dir: str = "./outputs"
    saver: str | None = "hdf5"
    logging: LoggingConfig | None = None

    tissue_detection: StageConfig | None = None
    artifact_detection: StageConfig | None = None
    statistics: StageConfig | None = None

    @property
    def input_dir(self) -> str:
        return self.slides.input_dir

    @property
    def file_list(self) -> str | list[str] | None:
        return self.slides.file_list

    @property
    def pattern(self) -> str:
        return self.slides.pattern

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CohortConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CohortConfig":
        data = data or {}

        slides = cls._build_slides_config(data.get("slides", {}))

        output_dir = data.get("output_dir", "./outputs")
        saver = data.get("saver", "hdf5")

        logging = cls._build_logging_config(
            data.get("logging", {}),
            output_dir=output_dir,
        )

        stages = data.get("stages", {}) or {}

        common_stage_kwargs = {
            "input_dir": slides.input_dir,
            "file_list": slides.file_list,
            "pattern": slides.pattern,
            "output_dir": output_dir,
            "saver": saver,
            "logging": logging,
        }

        tissue_detection = cls._build_stage_config(
            stages.get("tissue_detection"),
            stage_name="tissue_detection",
            **common_stage_kwargs,
        )

        artifact_detection = cls._build_stage_config(
            stages.get("artifact_detection"),
            stage_name="artifact_detection",
            **common_stage_kwargs,
        )

        statistics = cls._build_stage_config(
            stages.get("statistics"),
            stage_name="statistics",
            **common_stage_kwargs,
        )

        return cls(
            slides=slides,
            output_dir=output_dir,
            saver=saver,
            logging=logging,
            tissue_detection=tissue_detection,
            artifact_detection=artifact_detection,
            statistics=statistics,
        )

    @staticmethod
    def _build_slides_config(data: dict[str, Any]) -> SlidesConfig:
        data = data or {}

        return SlidesConfig(
            input_dir=data.get("input_dir", "./data/slides"),
            file_list=data.get("file_list"),
            pattern=data.get("pattern", "*.svs"),
        )

    @staticmethod
    def _build_logging_config(
        data: dict[str, Any],
        output_dir: str,
    ) -> LoggingConfig:
        data = data or {}

        return LoggingConfig(
            enabled=data.get("enabled", True),
            log_dir=data.get("log_dir", str(Path(output_dir) / "logs")),
            processed_file=data.get("processed_file", "processed.csv"),
            errors_file=data.get("errors_file", "errors.csv"),
        )

    @classmethod
    def _build_stage_config(
        cls,
        data: dict[str, Any] | None,
        stage_name: str,
        input_dir: str,
        file_list: str | list[str] | None,
        pattern: str,
        output_dir: str,
        saver: str | None,
        logging: LoggingConfig,
    ) -> StageConfig | None:
        if data is None:
            return None

        data = data or {}

        enabled = data.get("enabled", True)

        if not enabled:
            return None

        tissue_source = cls._build_source_config(data.get("tissue_source"))
        artifact_source = cls._build_source_config(data.get("artifact_source"))

        parallel_workers = data.get(
            "parallel_workers",
            data.get("workers", 1),
        )

        return StageConfig(
            enabled=enabled,
            algorithm=data.get("algorithm"),
            config_path=data.get("config_path"),
            parallel_workers=parallel_workers,
            overwrite=data.get("overwrite", False),
            tissue_source=tissue_source,
            artifact_source=artifact_source,
            input_dir=input_dir,
            file_list=file_list,
            pattern=pattern,
            output_dir=output_dir,
            saver=saver,
            logging=logging,
        )

    @staticmethod
    def _build_source_config(
        data: dict[str, Any] | None,
    ) -> SourceConfig | None:
        if data is None:
            return None

        if isinstance(data, SourceConfig):
            return data

        algorithm = data.get("algorithm")

        if algorithm is None:
            raise ValueError(
                f"Source config must contain 'algorithm'. Got: {data}"
            )

        return SourceConfig(
            algorithm=algorithm,
        )





