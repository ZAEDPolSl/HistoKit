import pytest
from histokit.savers import HDF5Saver, NoOpSaver, PickleSaver, Saver

def test_saver_none_uses_noop_saver():
    saver = Saver(method=None)

    assert isinstance(saver.saver, NoOpSaver)


@pytest.mark.parametrize(
    "method",
    [
        "none",
        "noop",
        "no_op",
        "null",
        "NONE",
        "NOOP",
    ],
)
def test_saver_noop_methods_use_noop_saver(method):
    saver = Saver(method=method)

    assert isinstance(saver.saver, NoOpSaver)


def test_noop_saver_save_does_nothing(tmp_path):
    saver = NoOpSaver()

    result = {
        "mask": [1, 2, 3],
    }

    saver.save(
        out_dir=tmp_path,
        basename="test",
        result=result,
    )

    assert list(tmp_path.iterdir()) == []


def test_noop_saver_load_raises():
    saver = NoOpSaver()

    with pytest.raises(RuntimeError, match="NoOpSaver does not support loading"):
        saver.load("dummy_path")


def test_saver_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown saver method"):
        Saver(method="unknown")


def test_saver_hdf5_method_creates_hdf5_saver():
    saver = Saver(method="hdf5")

    assert isinstance(saver.saver, HDF5Saver)


def test_saver_pickle_method_creates_pickle_saver():

    saver = Saver(method="pickle")

    assert isinstance(saver.saver, PickleSaver)