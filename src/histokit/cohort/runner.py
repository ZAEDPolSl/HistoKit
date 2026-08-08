from __future__ import annotations
from pathlib import Path
from itertools import product
from copy import deepcopy
import re
import yaml

from .config import CohortConfig
from .pipelines.tissue import TissueDetectionPipeline
from .pipelines.artifact import ArtifactDetectionPipeline


class CohortRunner:
    def __init__(self, config: CohortConfig):
        self.config = config

    def run(self):
        if self.config.tissue_detection is not None:
            print("=> Running tissue detection pipeline...")
            print(f"Input directory: {self.config.input_dir}")
            print(f"Output directory: {self.config.output_dir}")
            tissue_pipeline = TissueDetectionPipeline(
                self.config.tissue_detection
            )
            tissue_pipeline.run()

        if self.config.artifact_detection is not None:
            print("=> Running artifact detection pipeline...")
            print(f"Input directory: {self.config.input_dir}")
            print(f"Output directory: {self.config.output_dir}")
            artifact_pipeline = ArtifactDetectionPipeline(
                self.config.artifact_detection
            )
            artifact_pipeline.run()

        if self.config.statistics is not None:
            print("=> Calculating statistics...")
            print(f"Input directory: {self.config.input_dir}")
            print(f"Output directory: {self.config.output_dir}")
            raise NotImplementedError(
                "Statistics pipeline is not implemented yet."
            )
        
    @staticmethod
    def load_grid_search_config(path_or_dict):
        if isinstance(path_or_dict, (str, Path)):
            with open(path_or_dict, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        return path_or_dict or {}

    @staticmethod
    def iter_param_combinations(params: dict):
        keys = list(params.keys())

        values = [
            value if isinstance(value, list) else [value]
            for value in params.values()
        ]

        for combination in product(*values):
            yield dict(zip(keys, combination))

    def iter_grid_cases(self, grid_config: dict):
        for case in grid_config.get("cases", []):
            case_name = case["name"]
            params = case.get("params", {})

            for combination in self.iter_param_combinations(params):
                yield case_name, combination

    @staticmethod
    def safe_name(value):
        text = str(value).replace(".", "p")
        text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text)
        return text

    def make_run_name(self, params: dict):
        return "__".join(
            f"{key}_{self.safe_name(value)}"
            for key, value in params.items()
        )
        
    def run_grid_search(self, param_grid: dict | str | Path):
        param_grid = self.load_grid_search_config(param_grid)

        if self.config.tissue_detection is not None:
            print("=> Running tissue detection ...")
            print(f"Input directory: {self.config.input_dir}")
            print(f"Output directory: {self.config.output_dir}")

            tissue_pipeline = TissueDetectionPipeline(self.config.tissue_detection)
            tissue_pipeline.run()

        if self.config.artifact_detection is not None:
            artifact_grid = param_grid.get("artifact_detection", {})
            artifact_stage_config = deepcopy(self.config.artifact_detection)


            with open(Path(artifact_stage_config.config_path), "r", encoding="utf-8") as f:
                base_grandqc_config = yaml.safe_load(f)

            for case_name, params in self.iter_grid_cases(artifact_grid):
                grandqc_config = deepcopy(base_grandqc_config)
                stage_config = deepcopy(artifact_stage_config)

                for param, value in params.items():
                    grandqc_config[param] = value

                if grandqc_config.get("blending_mode") == "constant":
                    grandqc_config.pop("blending_sigma", None)

                run_name = self.make_run_name(params)

                output_dir = (
                    Path(self.config.output_dir)
                    / "grid_search"
                    / run_name
                )   

                output_dir.mkdir(parents=True, exist_ok=True)
                stage_config.output_dir = str(output_dir)

                temp_config_path = output_dir / "grandqc_grid_config.yaml"

                with open(temp_config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(grandqc_config, f, sort_keys=False)

                stage_config.config_path = str(temp_config_path)

                print("")
                print(f"=> Running artifact detection grid search: {case_name}")
                print(f"Parameters: {params}")

                artifact_pipeline = ArtifactDetectionPipeline(stage_config)
                artifact_pipeline.run()


    