import pytest

from histokit.segmentation.collectors.base import CompositeOutputCollector
from histokit.segmentation.collectors.image import (
    ThumbnailCollector,
    SegmentationOverlayCollector,
    HistogramCollector,
    ImageOutputCollector,
)
from histokit.segmentation.postprocessing.step import (
    Opening,
    FillHoles,
    RemoveSmallRegions,
)
from histokit.segmentation.tissue.gamred.config import GaMRedConfig


def test_default_config_builds_postprocess_steps():
    config = GaMRedConfig()

    assert len(config.postprocess_steps) == 4
    assert isinstance(config.postprocess_steps[0], Opening)
    assert isinstance(config.postprocess_steps[1], FillHoles)
    assert isinstance(config.postprocess_steps[2], Opening)
    assert isinstance(config.postprocess_steps[3], RemoveSmallRegions)


def test_default_config_builds_default_collectors_config():
    config = GaMRedConfig()

    assert config.collectors == [
        {"name": "ThumbnailCollector"},
        {"name": "SegmentationOverlayCollector"},
        {"name": "HistogramCollector"},
    ]


def test_build_output_collector_requires_out_dir():
    config = GaMRedConfig(out_dir=None)

    with pytest.raises(ValueError, match="out_dir must be provided"):
        config.build_output_collector()


def test_build_output_collector_from_defaults(tmp_path):
    config = GaMRedConfig(out_dir=tmp_path)

    collector = config.build_output_collector()

    assert isinstance(collector, CompositeOutputCollector)
    assert len(collector.collectors) == 3
    assert isinstance(collector.collectors[0], ThumbnailCollector)
    assert isinstance(collector.collectors[1], SegmentationOverlayCollector)
    assert isinstance(collector.collectors[2], HistogramCollector)


def test_build_output_collector_from_string_config(tmp_path):
    config = GaMRedConfig(
        out_dir=tmp_path,
        collectors=[
            "ImageOutputCollector",
        ],
    )

    collector = config.build_output_collector()

    assert isinstance(collector, CompositeOutputCollector)
    assert len(collector.collectors) == 1
    assert isinstance(collector.collectors[0], ImageOutputCollector)


def test_build_output_collector_from_dict_config_with_params(tmp_path):
    config = GaMRedConfig(
        out_dir=tmp_path,
        collectors=[
            {
                "name": "HistogramCollector",
                "params": {
                    "width": 800,
                    "height_per_channel": 200,
                },
            },
        ],
    )

    collector = config.build_output_collector()

    hist_collector = collector.collectors[0]

    assert isinstance(hist_collector, HistogramCollector)
    assert hist_collector.width == 800
    assert hist_collector.height_per_channel == 200


def test_build_output_collector_unknown_collector_raises(tmp_path):
    config = GaMRedConfig(
        out_dir=tmp_path,
        collectors=[
            {"name": "UnknownCollector"},
        ],
    )

    with pytest.raises(ValueError, match="Unknown output collector"):
        config.build_output_collector()


def test_build_output_collector_invalid_item_raises(tmp_path):
    config = GaMRedConfig(
        out_dir=tmp_path,
        collectors=[
            123,
        ],
    )

    with pytest.raises(TypeError, match="Invalid collector config"):
        config.build_output_collector()


def test_to_hdf5_dict_contains_expected_keys():
    config = GaMRedConfig()

    data = config.to_hdf5_dict()

    assert data["tissdet_mag"] == config.tissdet_mag
    assert data["thr_min"] == config.thr_min
    assert data["remove_green_pen"] == config.remove_green_pen
    assert data["remove_black_pen"] == config.remove_black_pen
    assert data["remove_gray_stains"] == config.remove_gray_stains
    assert data["fill_holes"] == config.fill_holes
    assert data["open_disk_radius"] == config.open_disk_radius
    assert data["remove_small_regions"] == config.remove_small_regions
    assert data["small_regions_thr"] == config.small_regions_thr
    assert data["collectors"] == config.collectors

    assert "postprocess_steps" in data
    assert len(data["postprocess_steps"]) == 4


def test_to_algorithm_dict():
    config = GaMRedConfig()

    data = config.to_algorithm_dict()

    assert data["name"] == "GaMRed"
    assert data["config"] == config.to_hdf5_dict()


def test_from_dict_ignores_unknown_fields():
    config = GaMRedConfig.from_dict(
        {
            "tissdet_mag": 5.0,
            "unknown_field": "ignored",
        }
    )

    assert config.tissdet_mag == 5.0
    assert not hasattr(config, "unknown_field")


def test_from_dict_builds_postprocess_steps():
    config = GaMRedConfig.from_dict(
        {
            "postprocess_steps": [
                {
                    "name": "Opening",
                    "params": {"disk_radius": 4},
                },
                {
                    "name": "FillHoles",
                    "params": {"enabled": False},
                },
                {
                    "name": "RemoveSmallRegions",
                    "params": {"thr_area": 100},
                },
            ]
        }
    )

    assert len(config.postprocess_steps) == 3
    assert isinstance(config.postprocess_steps[0], Opening)
    assert config.postprocess_steps[0].disk_radius == 4
    assert isinstance(config.postprocess_steps[1], FillHoles)
    assert config.postprocess_steps[1].enabled is False
    assert isinstance(config.postprocess_steps[2], RemoveSmallRegions)
    assert config.postprocess_steps[2].thr_area == 100


def test_from_dict_unknown_postprocess_step_raises():
    with pytest.raises(ValueError, match="Unknown postprocess step"):
        GaMRedConfig.from_dict(
            {
                "postprocess_steps": [
                    {"name": "UnknownStep"},
                ]
            }
        )


def test_from_yaml(tmp_path):
    yaml_path = tmp_path / "gamred.yaml"

    yaml_path.write_text(
        """
tissdet_mag: 5.0
vis_mag: 0.625
out_dir: results
collectors:
  - name: ImageOutputCollector
postprocess_steps:
  - name: Opening
    params:
      disk_radius: 3
  - name: FillHoles
    params:
      enabled: false
""",
        encoding="utf-8",
    )

    config = GaMRedConfig.from_yaml(yaml_path)

    assert config.tissdet_mag == 5.0
    assert config.vis_mag == 0.625
    assert config.out_dir == "results"

    assert config.collectors == [
        {"name": "ImageOutputCollector"},
    ]

    assert len(config.postprocess_steps) == 2
    assert isinstance(config.postprocess_steps[0], Opening)
    assert config.postprocess_steps[0].disk_radius == 3
    assert isinstance(config.postprocess_steps[1], FillHoles)
    assert config.postprocess_steps[1].enabled is False