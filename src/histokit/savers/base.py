from abc import ABC, abstractmethod
from pathlib import Path

class BaseSaver(ABC):
    extension: str | None = None

    @abstractmethod
    def save(self, out_dir: str, basename: str, result: dict):
        raise NotImplementedError

    @abstractmethod
    def load(self, path) -> dict:
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

    @property
    def extension(self) -> str | None:
        return self.saver.extension

    def make_path(self, out_dir: str, basename: str):
        if self.extension is None:
            return None

        return Path(out_dir) / f"{basename}{self.extension}"

    def save(self, out_dir: str, basename: str, result: dict):
        self.saver.save(out_dir, basename, result)

    def load(self, path) -> dict:
        return self.saver.load(path)
    

class NoOpSaver(BaseSaver):

    def save(self, out_dir: str, basename: str, result: dict):
        pass

    def load(self, path) -> dict:
        raise RuntimeError("NoOpSaver does not support loading.")