from abc import ABC, abstractmethod

class BaseSaver(ABC):

    @abstractmethod
    def save(self, out_dir: str, basename:str, result: dict):
        raise NotImplementedError


class Saver:

    def __init__(self, method=None):

        if method is None:
            self.saver = NoOpSaver()
            return

        method = method.lower()

        if method in {"none", "noop", "no_op", "null"}:
            self.saver = NoOpSaver()

        elif method == "hdf5":
            from .hdf5 import HDF5Saver
            self.saver = HDF5Saver()

        elif method == "pickle":
            from .pickle import PickleSaver
            self.saver = PickleSaver()

        else:
            raise ValueError(f"Unknown saver method: {method}")

    def save(self, out_dir: str, basename: str, result: dict):
        self.saver.save(out_dir, basename, result)

    def load(self, mask_path) -> dict:
        return self.saver.load(mask_path)
    

class NoOpSaver(BaseSaver):

    def save(self, out_dir: str, basename: str, result: dict):
        pass

    def load(self, path) -> dict:
        raise RuntimeError("NoOpSaver does not support loading.")