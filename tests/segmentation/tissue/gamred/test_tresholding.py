import numpy as np
import pytest

from histokit.segmentation.tissue.gamred.thresholding import (
    otsuthresh,
    two_step_otsu,
    get_thr_image,
)


def test_otsuthresh_returns_values_between_zero_and_one():
    counts = np.zeros(256)
    counts[50] = 100
    counts[200] = 100

    threshold, effectiveness = otsuthresh(counts)

    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= effectiveness <= 1.0


def test_otsuthresh_returns_zero_for_empty_histogram():
    counts = np.zeros(256)

    threshold, effectiveness = otsuthresh(counts)

    assert threshold == 0.0
    assert effectiveness == 0.0


def test_otsuthresh_detects_separation_for_bimodal_histogram():
    counts = np.zeros(256)
    counts[40:60] = 100
    counts[180:200] = 100

    threshold, effectiveness = otsuthresh(counts)

    assert 0.1 < threshold < 0.9
    assert effectiveness > 0.0


def test_two_step_otsu_returns_valid_threshold():
    hist = np.zeros(256)
    hist[40:60] = 100
    hist[180:200] = 100

    threshold = two_step_otsu(hist)

    assert 0 <= threshold <= 255


def test_two_step_otsu_returns_number():
    hist = np.ones(256)

    threshold = two_step_otsu(hist)

    assert np.isscalar(threshold)
    assert np.isfinite(threshold)


def test_get_thr_image_returns_expected_keys_and_histograms():
    img = np.zeros((20, 20, 3), dtype=np.uint8)

    img[..., 0] = 50
    img[..., 1] = 100
    img[..., 2] = 150

    thr, R, G, B = get_thr_image(
        img,
        thr_min=0,
    )

    assert set(thr.keys()) == {"R", "G", "B"}

    assert R.shape == (256,)
    assert G.shape == (256,)
    assert B.shape == (256,)

    assert R[50] == 400
    assert G[100] == 400
    assert B[150] == 400


def test_get_thr_image_thresholds_are_finite():
    img = np.zeros((20, 20, 3), dtype=np.uint8)

    img[:10, :, :] = 50
    img[10:, :, :] = 200

    thr, _, _, _ = get_thr_image(
        img,
        thr_min=0,
    )

    assert np.isfinite(thr["R"])
    assert np.isfinite(thr["G"])
    assert np.isfinite(thr["B"])


def test_get_thr_image_uses_otsu_when_threshold_below_min(monkeypatch):
    import histokit.segmentation.tissue.gamred.thresholding as thresholding

    def fake_gamred_hist(x, hist):
        return 10, 0, {}

    def fake_two_step_otsu(hist):
        return 123

    monkeypatch.setattr(
        thresholding,
        "GaMRed_hist",
        fake_gamred_hist,
    )

    monkeypatch.setattr(
        thresholding,
        "two_step_otsu",
        fake_two_step_otsu,
    )

    img = np.zeros((10, 10, 3), dtype=np.uint8)

    thr, _, _, _ = thresholding.get_thr_image(
        img,
        thr_min=100,
    )

    assert thr == {
        "R": 123,
        "G": 123,
        "B": 123,
    }


def test_get_thr_image_does_not_use_otsu_when_threshold_above_min(monkeypatch):
    import histokit.segmentation.tissue.gamred.thresholding as thresholding

    def fake_gamred_hist(x, hist):
        return 200, 0, {}

    def fake_two_step_otsu(hist):
        raise AssertionError("two_step_otsu should not be called")

    monkeypatch.setattr(
        thresholding,
        "GaMRed_hist",
        fake_gamred_hist,
    )

    monkeypatch.setattr(
        thresholding,
        "two_step_otsu",
        fake_two_step_otsu,
    )

    img = np.zeros((10, 10, 3), dtype=np.uint8)

    thr, _, _, _ = thresholding.get_thr_image(
        img,
        thr_min=100,
    )

    assert thr == {
        "R": 200,
        "G": 200,
        "B": 200,
    }


def test_get_thr_image_verbose_prints_message(monkeypatch, capsys):
    import histokit.segmentation.tissue.gamred.thresholding as thresholding

    def fake_gamred_hist(x, hist):
        return 10, 0, {}

    def fake_two_step_otsu(hist):
        return 123

    monkeypatch.setattr(
        thresholding,
        "GaMRed_hist",
        fake_gamred_hist,
    )

    monkeypatch.setattr(
        thresholding,
        "two_step_otsu",
        fake_two_step_otsu,
    )

    img = np.zeros((10, 10, 3), dtype=np.uint8)

    thresholding.get_thr_image(
        img,
        thr_min=100,
        verbose=True,
    )

    captured = capsys.readouterr()

    assert "Too low threshold for R channel" in captured.out
    assert "Too low threshold for G channel" in captured.out
    assert "Too low threshold for B channel" in captured.out