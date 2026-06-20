import os
import pickle

from .base import BaseSaver


class PickleSaver(BaseSaver):

    def save(
        self,
        out_dir: str,
        basename: str,
        result: dict,
    ) -> None:

        with open(
            os.path.join(out_dir, f"{basename}.pkl"),
            "wb",
        ) as f:
            pickle.dump(
                result,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path) -> dict:

        with open(path, "rb") as f:
            return pickle.load(f)