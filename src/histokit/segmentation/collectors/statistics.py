from pathlib import Path
import csv
import os
import numpy as np
from .base import OutputCollector, PipelineOutput


class ArtifactStatisticsCollector(OutputCollector):
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "artifact_statistics.csv"

    def emit(self, output: PipelineOutput) -> None:
        
        if output.name != "artifact_statistics":
            return
        
        basename = output.metadata.get("basename", "overlay")
        image = np.asarray(output.metadata["image"])
        mask = np.asarray(output.data)
        colors = output.metadata.get("colors", {np.unique(mask)[i]: f"color_{i}" for i in range(len(np.unique(mask)))})

        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        else:
            mask = mask.astype(np.uint8)

        cols = ["basename", "color", "area", "percentage"]
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)

            if not self.csv_path.exists():
                writer.writeheader()

            for i in range(10):
                nowy_wiersz = {
                    "imie": f"Osoba_{i}",
                    "wiek": 20 + i,
                    "miasto": "Warszawa"
                }

                writer.writerow(nowy_wiersz)

        save_dir = self.out_dir / "artifact_statistics"
        save_dir.mkdir(parents=True, exist_ok=True)

