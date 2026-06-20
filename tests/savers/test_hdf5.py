import numpy as np
import pytest
from histokit.savers import HDF5Saver



def test_save_and_load_segmentation_result(tmp_path):

    saver = HDF5Saver()

    result = {
        "basename": "slide_001",
        "method": "GaMRed",
        "time": 1.23,
        "mask": [
            np.array([[0, 1], [1, 0]], dtype=np.uint8),
            np.array([[1, 1], [0, 0]], dtype=np.uint8),
        ],
        "bbox": [
            [0, 0, 100, 100],
            [100, 100, 200, 200],
        ],
        "config": {
            "thr_min": 178.5,
            "remove_pen": True,
        },
    }

    saver.save(
        out_dir=tmp_path,
        basename="test",
        result=result,
    )

    loaded = saver.load(
        tmp_path / "test.h5"
    )

    assert loaded["basename"] == result["basename"]
    assert loaded["method"] == result["method"]
    assert loaded["time"] == result["time"]

    np.testing.assert_array_equal(
        loaded["mask"][0],
        result["mask"][0],
    )

    np.testing.assert_array_equal(
        loaded["mask"][1],
        result["mask"][1],
    )

    assert loaded["config"] == result["config"]


def test_save_and_load_supported_types(tmp_path):

    saver = HDF5Saver()

    result = {
        "string": "abc",
        "int": 5,
        "float": 3.14,
        "bool": True,
        "none": None,
        "tuple": (1, 2, 3),
        "array": np.array([1, 2, 3]),
        "dict": {
            "nested": "value",
        },
        "list": [
            1,
            2,
            3,
        ],
    }

    saver.save(
        tmp_path,
        "types",
        result,
    )

    loaded = saver.load(
        tmp_path / "types.h5"
    )

    assert loaded["string"] == "abc"
    assert loaded["int"] == 5
    assert loaded["float"] == 3.14
    assert loaded["bool"] is True
    assert loaded["none"] is None

    np.testing.assert_array_equal(
        loaded["tuple"],
        np.array([1, 2, 3]),
    )

    np.testing.assert_array_equal(
        loaded["array"],
        np.array([1, 2, 3]),
    )

    assert loaded["dict"] == {
        "nested": "value",
    }

    assert loaded["list"] == [1, 2, 3]


def test_unsupported_type_raises(tmp_path):

    saver = HDF5Saver()

    result = {
        "bad": object(),
    }

    with pytest.raises(TypeError):
        saver.save(
            tmp_path,
            "bad",
            result,
        )