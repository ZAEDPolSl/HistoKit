import numpy as np
import pytest
from histokit.segmentation.collectors import PipelineOutput
from histokit.segmentation.tissue.gamred import GaMRedConfig, GaMRedSegmenter


class DummyCollector:
    def __init__(self):
        self.outputs = []

    def emit(self, output: PipelineOutput):
        self.outputs.append(output)


class DummySaver:
    def __init__(self):
        self.calls = []

    def save(self, out_dir, basename, result):
        self.calls.append(
            {
                "out_dir": out_dir,
                "basename": basename,
                "result": result,
            }
        )


class DummySlide:
    mag = 20
    mpp = 0.5
    level_dimensions = [(1000, 800)]

    def __init__(self):
        self.calls = []

    def read_region(self, mag):
        self.calls.append(mag)

        if mag == 2.5:
            img = np.full((20, 20, 3), 255, dtype=np.uint8)
            img[5:15, 5:15] = [80, 40, 40]
            return img

        if mag == 1.0:
            return np.full((10, 10, 3), 128, dtype=np.uint8)

        return np.full((10, 10, 3), 0, dtype=np.uint8)


def test_collect_emits_pipeline_output(tmp_path):
    config = GaMRedConfig(
        out_dir=tmp_path,
        saver=None,
        collectors=[],
        remove_green_pen=False,
        remove_black_pen=False,
        remove_gray_stains=False,
        postprocess_steps=[],
    )

    segmenter = GaMRedSegmenter(config)
    collector = DummyCollector()
    segmenter.output_collector = collector

    data = np.zeros((5, 5), dtype=np.uint8)

    segmenter._collect(
        name="test_output",
        step="test_step",
        kind="mask",
        data=data,
        basename="slide_001",
    )

    assert len(collector.outputs) == 1

    output = collector.outputs[0]

    assert output.name == "test_output"
    assert output.step == "test_step"
    assert output.kind == "mask"
    assert output.data is data
    assert output.metadata == {
        "basename": "slide_001",
    }


def test_segment_returns_expected_result_keys(monkeypatch, tmp_path):
    import histokit.segmentation.tissue.gamred.segmenter as segmenter_module

    def fake_get_thr_image(region_np, thr_min):
        hist = np.ones(256, dtype=np.int64)
        return {"R": 200, "G": 200, "B": 200}, hist, hist, hist

    def fake_rescale_mask(mask, scale):
        return mask

    def fake_split_regions(mask):
        return [mask], [[0, 0, mask.shape[1], mask.shape[0]]]

    monkeypatch.setattr(segmenter_module, "get_thr_image", fake_get_thr_image)
    monkeypatch.setattr(segmenter_module, "rescale_mask", fake_rescale_mask)
    monkeypatch.setattr(segmenter_module, "split_regions", fake_split_regions)

    config = GaMRedConfig(
        out_dir=tmp_path,
        saver=None,
        collectors=[],
        remove_green_pen=False,
        remove_black_pen=False,
        remove_gray_stains=False,
        postprocess_steps=[],
    )

    segmenter = GaMRedSegmenter(config)
    segmenter.output_collector = DummyCollector()
    segmenter.saver = DummySaver()

    slide = DummySlide()

    result = segmenter.segment(
        slide=slide,
        basename="slide_001",
    )

    expected_keys = {
        "basename",
        "method",
        "type",
        "mask",
        "bbox",
        "mag_det",
        "mag_save",
        "mag_l0",
        "mpp_l0",
        "level_dimensions_0",
        "thr",
        "config",
        "time",
    }

    assert set(result.keys()) == expected_keys
    assert result["basename"] == "slide_001"
    assert result["method"] == "GaMRed"
    assert result["type"] == "tissue_mask"
    assert result["mag_det"] == config.tissdet_mag
    assert result["mag_save"] == config.save_mag
    assert result["mag_l0"] == slide.mag
    assert result["mpp_l0"] == slide.mpp

    np.testing.assert_array_equal(
        result["level_dimensions_0"],
        np.array(slide.level_dimensions[0]),
    )


def test_segment_collects_expected_outputs(monkeypatch, tmp_path):
    import histokit.segmentation.tissue.gamred.segmenter as segmenter_module

    def fake_get_thr_image(region_np, thr_min):
        hist = np.ones(256, dtype=np.int64)
        return {"R": 200, "G": 200, "B": 200}, hist, hist, hist

    def fake_rescale_mask(mask, scale):
        return mask

    def fake_split_regions(mask):
        return [mask], [[0, 0, mask.shape[1], mask.shape[0]]]

    monkeypatch.setattr(segmenter_module, "get_thr_image", fake_get_thr_image)
    monkeypatch.setattr(segmenter_module, "rescale_mask", fake_rescale_mask)
    monkeypatch.setattr(segmenter_module, "split_regions", fake_split_regions)

    config = GaMRedConfig(
        out_dir=tmp_path,
        saver=None,
        collectors=[],
        remove_green_pen=False,
        remove_black_pen=False,
        remove_gray_stains=False,
        postprocess_steps=[],
    )

    segmenter = GaMRedSegmenter(config)

    collector = DummyCollector()
    segmenter.output_collector = collector
    segmenter.saver = DummySaver()

    segmenter.segment(
        slide=DummySlide(),
        basename="slide_001",
    )

    names = [output.name for output in collector.outputs]
    steps = [output.step for output in collector.outputs]

    assert names == [
        "thumbnail",
        "tissue_overlay",
        "histograms",
    ]

    assert steps == [
        "input",
        "visualisation",
        "thresholding",
    ]


def test_segment_calls_saver(monkeypatch, tmp_path):
    import histokit.segmentation.tissue.gamred.segmenter as segmenter_module

    def fake_get_thr_image(region_np, thr_min):
        hist = np.ones(256, dtype=np.int64)
        return {"R": 200, "G": 200, "B": 200}, hist, hist, hist

    def fake_rescale_mask(mask, scale):
        return mask

    def fake_split_regions(mask):
        return [mask], [[0, 0, mask.shape[1], mask.shape[0]]]

    monkeypatch.setattr(segmenter_module, "get_thr_image", fake_get_thr_image)
    monkeypatch.setattr(segmenter_module, "rescale_mask", fake_rescale_mask)
    monkeypatch.setattr(segmenter_module, "split_regions", fake_split_regions)

    config = GaMRedConfig(
        out_dir=tmp_path,
        saver=None,
        collectors=[],
        remove_green_pen=False,
        remove_black_pen=False,
        remove_gray_stains=False,
        postprocess_steps=[],
    )

    segmenter = GaMRedSegmenter(config)
    segmenter.output_collector = DummyCollector()

    saver = DummySaver()
    segmenter.saver = saver

    result = segmenter.segment(
        slide=DummySlide(),
        basename="slide_001",
    )

    assert len(saver.calls) == 1
    assert saver.calls[0]["out_dir"] == tmp_path
    assert saver.calls[0]["basename"] == "slide_001"
    assert saver.calls[0]["result"] is result


def test_segment_applies_postprocess_steps(monkeypatch, tmp_path):
    import histokit.segmentation.tissue.gamred.segmenter as segmenter_module

    class DummyStep:
        def __init__(self):
            self.called = False

        def __call__(self, mask):
            self.called = True
            return mask

        def get_config(self):
            return {"name": "DummyStep", "params": {}}

    def fake_get_thr_image(region_np, thr_min):
        hist = np.ones(256, dtype=np.int64)
        return {"R": 200, "G": 200, "B": 200}, hist, hist, hist

    def fake_rescale_mask(mask, scale):
        return mask

    def fake_split_regions(mask):
        return [mask], [[0, 0, mask.shape[1], mask.shape[0]]]

    monkeypatch.setattr(segmenter_module, "get_thr_image", fake_get_thr_image)
    monkeypatch.setattr(segmenter_module, "rescale_mask", fake_rescale_mask)
    monkeypatch.setattr(segmenter_module, "split_regions", fake_split_regions)

    step = DummyStep()

    config = GaMRedConfig(
        out_dir=tmp_path,
        saver=None,
        collectors=[],
        remove_green_pen=False,
        remove_black_pen=False,
        remove_gray_stains=False,
        postprocess_steps=[step],
    )

    segmenter = GaMRedSegmenter(config)
    segmenter.output_collector = DummyCollector()
    segmenter.saver = DummySaver()

    segmenter.segment(
        slide=DummySlide(),
        basename="slide_001",
    )

    assert step.called