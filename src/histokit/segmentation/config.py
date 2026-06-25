from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import yaml

from .collectors.base import (
    CompositeOutputCollector,
    NoOpOutputCollector,
)


CollectorConfig = dict[str, Any] | str


@dataclass
class BaseAlgorithmConfig:
    saver: str | None = "hdf5"
    out_dir: str | Path | None = "./outputs"
    save_mag: float | None = 1.0

    collectors: list[CollectorConfig] | None = None

    algorithm_name: ClassVar[str] = "Base"
    collector_registry: ClassVar[dict[str, type]] = {}
    default_collectors: ClassVar[list[CollectorConfig]] = []

    def __post_init__(self) -> None:
        # None means: use algorithm defaults.
        # [] means: collect nothing.
        if self.collectors is None:
            self.collectors = copy.deepcopy(self.default_collectors)

    def build_output_collector(
        self,
        out_dir: str | Path | None = None,
    ):
        if not self.collectors:
            return NoOpOutputCollector()

        out_dir = out_dir if out_dir is not None else self.out_dir

        if out_dir is None:
            raise ValueError("out_dir must be provided to build collectors.")

        collector_instances = []

        for item in self.collectors:
            name, params = self._parse_collector_config(item)

            try:
                collector_cls = self.collector_registry[name]
            except KeyError:
                raise ValueError(
                    f"Unknown output collector: {name}. "
                    f"Available collectors: {list(self.collector_registry)}"
                )

            collector_instances.append(
                collector_cls(
                    out_dir=out_dir,
                    **params,
                )
            )

        return CompositeOutputCollector(collector_instances)

    @staticmethod
    def _parse_collector_config(
        item: CollectorConfig,
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(item, str):
            return item, {}

        if isinstance(item, dict):
            name = item.get("name")
            params = item.get("params", {})

            if name is None:
                raise ValueError(f"Collector config is missing 'name': {item}")

            return name, params

        raise TypeError(f"Invalid collector config: {item}")

    def common_hdf5_dict(self) -> dict[str, Any]:
        return {
            "saver": self.saver,
            "out_dir": str(self.out_dir) if self.out_dir is not None else None,
            "save_mag": self.save_mag,
            "collectors": self.collectors,
        }

    def to_hdf5_dict(self) -> dict[str, Any]:
        return self.common_hdf5_dict()

    def to_algorithm_dict(self) -> dict[str, Any]:
        return {
            "name": self.algorithm_name,
            "config": self.to_hdf5_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaseAlgorithmConfig":
        data = dict(data or {})
        data = cls._preprocess_dict(data)

        field_names = {f.name for f in fields(cls)}

        filtered = {
            k: v
            for k, v in data.items()
            if k in field_names
        }

        return cls(**filtered)

    @classmethod
    def _preprocess_dict(cls, data: dict) -> dict:
        return data

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        return cls.from_dict(data)