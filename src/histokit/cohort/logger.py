import csv
import traceback
from pathlib import Path


class CohortLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.processed_path = self.log_dir / "processed.csv"
        self.errors_path = self.log_dir / "errors.csv"

        self._init_file(
            self.processed_path,
            ["stage", "slide_path", "basename", "status", "time"],
        )

        self._init_file(
            self.errors_path,
            ["stage", "slide_path", "basename", "error", "traceback"],
        )

    def _init_file(self, path, header):
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def log_processed(self, stage, slide_path, basename, elapsed):
        with open(self.processed_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [stage, str(slide_path), basename, "success", elapsed]
            )

    def log_error(self, stage, slide_path, basename, error):
        with open(self.errors_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    stage,
                    str(slide_path),
                    basename,
                    repr(error),
                    traceback.format_exc(),
                ]
            )