from pathlib import Path
import argparse
from histokit import CohortConfig
from histokit import CohortRunner

DESCRIPTION = r"""
╔══════════════════════════════════════════════════════════════╗
║                         HistoKit                             ║
║                  Tissue & Artifact Segmentation              ║
╚══════════════════════════════════════════════════════════════╝

Run HistoKit cohort processing on whole-slide images.

This command runs selected HistoKit pipelines on a cohort of slides.
Currently supported stages include:
  - tissue detection
  - artifact detection
  - statistics calculation

The pipeline behavior is controlled by a YAML configuration file.
"""


EPILOG = """
Required YAML config
--------------------
Before running the command, prepare a cohort YAML file defining:
  - slides.input_dir
  - slides.pattern or slides.file_list
  - output_dir
  - saver
  - enabled stages under stages

Example
-------
  python run_cohort.py --config ./configs/cohort.yaml

Example YAML
------------
  slides:
    input_dir: ./data/slides
    file_list: null
    pattern: "*.svs"

  output_dir: ./outputs
  saver: hdf5

  stages:
    tissue_detection:
      enabled: true
      algorithm: gamred
      config_path: ./configs/gamred.yaml
      parallel_workers: 8
      overwrite: false

    artifact_detection:
      enabled: true
      algorithm: grandqc
      config_path: ./configs/grandqc.yaml
      parallel_workers: 1
      overwrite: false
      tissue_source:
        algorithm: gamred
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the cohort YAML configuration file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = CohortConfig.from_yaml(args.config)

    runner = CohortRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
