import pytest

from histokit.segmentation.collectors import (
    CompositeOutputCollector,
    NoOpOutputCollector,
    OutputCollector,
    OutputKind,
    PipelineOutput,
)


def test_output_kind_values():
    assert OutputKind.IMAGE.value == "image"
    assert OutputKind.MASK.value == "mask"
    assert OutputKind.HISTOGRAM.value == "histogram"
    assert OutputKind.METADATA.value == "metadata"


def test_pipeline_output_creation():
    output = PipelineOutput(
        name="mask",
        step="segmentation",
        kind=OutputKind.MASK,
        data=[1, 2, 3],
    )

    assert output.name == "mask"
    assert output.step == "segmentation"
    assert output.kind == OutputKind.MASK
    assert output.data == [1, 2, 3]
    assert output.metadata == {}


def test_pipeline_output_with_metadata():
    output = PipelineOutput(
        name="mask",
        step="segmentation",
        kind=OutputKind.MASK,
        data=[1, 2, 3],
        metadata={"slide": "TCGA_001"},
    )

    assert output.metadata == {
        "slide": "TCGA_001",
    }


def test_noop_collector_accepts_output():
    collector = NoOpOutputCollector()

    output = PipelineOutput(
        name="mask",
        step="segmentation",
        kind=OutputKind.MASK,
        data=None,
    )

    collector.emit(output)


def test_output_collector_is_abstract():
    collector = OutputCollector()

    output = PipelineOutput(
        name="mask",
        step="segmentation",
        kind=OutputKind.MASK,
        data=None,
    )

    with pytest.raises(NotImplementedError):
        collector.emit(output)


class DummyCollector(OutputCollector):

    def __init__(self):
        self.outputs = []

    def emit(self, output):
        self.outputs.append(output)


def test_composite_collector_forwards_output_to_all_collectors():
    collector1 = DummyCollector()
    collector2 = DummyCollector()

    composite = CompositeOutputCollector(
        [collector1, collector2]
    )

    output = PipelineOutput(
        name="mask",
        step="segmentation",
        kind=OutputKind.MASK,
        data=[1, 2, 3],
    )

    composite.emit(output)

    assert len(collector1.outputs) == 1
    assert len(collector2.outputs) == 1

    assert collector1.outputs[0] is output
    assert collector2.outputs[0] is output


def test_composite_collector_with_empty_list():
    composite = CompositeOutputCollector([])

    output = PipelineOutput(
        name="mask",
        step="segmentation",
        kind=OutputKind.MASK,
        data=None,
    )

    composite.emit(output)


def test_composite_collector_preserves_metadata():
    collector = DummyCollector()

    composite = CompositeOutputCollector([collector])

    output = PipelineOutput(
        name="mask",
        step="segmentation",
        kind=OutputKind.MASK,
        data=None,
        metadata={
            "basename": "slide_001",
            "mag": 2.5,
        },
    )

    composite.emit(output)

    assert collector.outputs[0].metadata == {
        "basename": "slide_001",
        "mag": 2.5,
    }