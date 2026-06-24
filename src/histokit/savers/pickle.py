import os
import pickle
from .base import BaseSaver


class PickleSaver(BaseSaver):
    """Saver class for saving and loading data using Pickle format."""

    def save(
        self,
        out_dir: str,
        basename: str,
        result: dict,
    ) -> None:
        
        """Save a dictionary to a Pickle file.
        param out_dir: Directory to save the Pickle file.
        param basename: Base name of the Pickle file (without extension).
        param result: Dictionary containing the data to save.
        """
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
        """Load a dictionary from a Pickle file.
        param path: Path to the Pickle file.
        return: Dictionary containing the data from the Pickle file.
        """
        with open(path, "rb") as f:
            return pickle.load(f)