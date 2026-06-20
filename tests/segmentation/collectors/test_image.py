import numpy as np
from PIL import Image

from histokit.segmentation.collectors.base import (
    OutputKind,
    PipelineOutput,
)
from histokit.segmentation.collectors.image import (
    ImageOutputCollector,
    ThumbnailCollector,
    SegmentationOverlayCollector,
    ArtifactOverlayCollector,
    HistogramCollector,
)


def test_image_output_collector_saves_image(tmp_path):
    collector = ImageOutputCollector(tmp_path)

    output = PipelineOutput(
        name="region",
        step="input",
        kind=OutputKind.IMAGE,
        data=np.zeros((10, 10, 3), dtype=np.uint8),
        metadata={"basename": "slide_001", "step": "input"},
    )

    collector.emit(output)

    assert (tmp_path / "input" / "slide_001.png").exists()


def test_image_output_collector_saves_bool_mask_as_png(tmp_path):
    collector = ImageOutputCollector(tmp_path)

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    output = PipelineOutput(
        name="mask",
        step="thresholding",
        kind=OutputKind.MASK,
        data=mask,
        metadata={"basename": "slide_001", "step": "thresholding"},
    )

    collector.emit(output)

    path = tmp_path / "thresholding" / "slide_001.png"

    assert path.exists()

    saved = np.array(Image.open(path))

    assert saved.max() == 255
    assert saved.min() == 0


def test_image_output_collector_ignores_metadata(tmp_path):
    collector = ImageOutputCollector(tmp_path)

    output = PipelineOutput(
        name="config",
        step="metadata",
        kind=OutputKind.METADATA,
        data={"a": 1},
    )

    collector.emit(output)

    assert list(tmp_path.iterdir()) == []


def test_thumbnail_collector_saves_thumbnail(tmp_path):
    collector = ThumbnailCollector(tmp_path)

    output = PipelineOutput(
        name="thumbnail",
        step="input",
        kind=OutputKind.IMAGE,
        data=np.zeros((10, 10, 3), dtype=np.uint8),
        metadata={"basename": "slide_001"},
    )

    collector.emit(output)

    assert (tmp_path / "thumbnails" / "slide_001.png").exists()


def test_thumbnail_collector_ignores_other_outputs(tmp_path):
    collector = ThumbnailCollector(tmp_path)

    output = PipelineOutput(
        name="region",
        step="input",
        kind=OutputKind.IMAGE,
        data=np.zeros((10, 10, 3), dtype=np.uint8),
        metadata={"basename": "slide_001"},
    )

    collector.emit(output)

    assert list(tmp_path.iterdir()) == []


def test_segmentation_overlay_collector_saves_overlay(tmp_path):
    collector = SegmentationOverlayCollector(tmp_path)

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((25, 25), dtype=np.uint8)
    mask[5:20, 5:20] = 255

    output = PipelineOutput(
        name="tissue_overlay",
        step="overlay",
        kind=OutputKind.MASK,
        data=mask,
        metadata={
            "basename": "slide_001",
            "image": image,
        },
    )

    collector.emit(output)

    assert (
        tmp_path
        / "tissue_detection_overlay"
        / "slide_001.png"
    ).exists()


def test_artifact_overlay_collector_saves_overlay_and_map(tmp_path):
    collector = ArtifactOverlayCollector(tmp_path)

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((25, 25), dtype=np.uint8)
    mask[5:20, 5:20] = 1

    output = PipelineOutput(
        name="artifact_overlay",
        step="overlay",
        kind=OutputKind.MASK,
        data=mask,
        metadata={
            "basename": "slide_001",
            "image": image,
            "colors": {
                "artifact": (1, (255, 0, 0)),
            },
        },
    )

    collector.emit(output)

    assert (
        tmp_path
        / "artifact_detection_overlay"
        / "slide_001.png"
    ).exists()

    assert (
        tmp_path
        / "artifact_detection_map"
        / "slide_001.png"
    ).exists()


def test_histogram_collector_saves_histogram(tmp_path):
    collector = HistogramCollector(tmp_path)

    hist = np.ones(256, dtype=np.int64)

    output = PipelineOutput(
        name="histograms",
        step="histograms",
        kind=OutputKind.HISTOGRAM,
        data={
            "R": hist,
            "G": hist,
            "B": hist,
            "thr": {
                "R": 100,
                "G": 120,
                "B": 140,
            },
        },
        metadata={"basename": "slide_001"},
    )

    collector.emit(output)

    assert (tmp_path / "histograms" / "slide_001.png").exists()