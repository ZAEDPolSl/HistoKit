from importlib import import_module

from .cohort import (
    SlidesConfig,
    CohortConfig,
    StageConfig,
    LoggingConfig,
    SourceConfig,
    CohortLogger,
    CohortRunner,
)

__all__ = [
    "augmentation",
    "patch_extractors",
    "savers",
    "segmentation",
    "slide",
    "stain_normalisation",
    "cohort",
    "SlidesConfig",
    "CohortConfig",
    "StageConfig",
    "LoggingConfig",
    "SourceConfig",
    "CohortLogger",
    "CohortRunner",
]


def __getattr__(name):
    if name in {
        "augmentation",
        "patch_extractors",
        "savers",
        "segmentation",
        "slide",
        "stain_normalisation",
        "cohort",
    }:
        return import_module(f".{name}", __name__)

    raise AttributeError(name)