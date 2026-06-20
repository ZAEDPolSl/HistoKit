import numpy as np
import pytest
from histokit.savers import PickleSaver


def test_pickle_saver_save_creates_file(tmp_path):
    saver = PickleSaver()

    result = {
        "basename": "slide_001",
        "method": "GaMRed",
    }

    saver.save(
        out_dir=tmp_path,
        basename="test",
        result=result,
    )

    assert (tmp_path / "test.pkl").exists()


def test_pickle_saver_save_and_load(tmp_path):
    saver = PickleSaver()

    result = {
        "basename": "slide_001",
        "method": "GaMRed",
        "time": 1.23,
        "mask": [
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
            np.array([[255, 255], [0, 0]], dtype=np.uint8),
        ],
        "bbox": [
            [0, 0, 100, 100],
            [100, 100, 200, 200],
        ],
        "config": {
            "thr_min": 178.5,
            "remove_green_pen": True,
            "thr_green_pen": (15, 120),
        },
    }

    saver.save(
        out_dir=tmp_path,
        basename="test",
        result=result,
    )

    loaded = saver.load(tmp_path / "test.pkl")

    assert loaded["basename"] == result["basename"]
    assert loaded["method"] == result["method"]
    assert loaded["time"] == result["time"]
    assert loaded["bbox"] == result["bbox"]
    assert loaded["config"] == result["config"]

    np.testing.assert_array_equal(
        loaded["mask"][0],
        result["mask"][0],
    )

    np.testing.assert_array_equal(
        loaded["mask"][1],
        result["mask"][1],
    )


def test_pickle_saver_load_missing_file_raises(tmp_path):
    saver = PickleSaver()

    with pytest.raises(FileNotFoundError):
        saver.load(tmp_path / "missing.pkl")