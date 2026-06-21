from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from ...savers.base import Saver
from ..logger import CohortLogger


class BaseCohortPipeline(ABC):
    stage_name: str
    result_subdir: str

    def __init__(self, config):
        self.config = config

    def result_saver(self):
        return Saver(getattr(self.config, "saver", "hdf5"))

    def stage_dir(
        self,
        algorithm: str | None = None,
    ) -> Path:
        algorithm = algorithm or self.config.algorithm

        return (
            Path(self.config.output_dir)
            / self.result_subdir
            / algorithm
        )

    def visualization_dir(
        self,
        algorithm: str | None = None,
    ) -> Path:
        return self.stage_dir(algorithm=algorithm) / "visualizations"

    def output_path(
        self,
        slide_path: Path,
        algorithm: str | None = None,
    ) -> Path:
        return (
            self.stage_dir(algorithm=algorithm)
            / f"{slide_path.stem}{self.result_saver().extension}"
        )

    def result_path(
        self,
        slide_path: Path,
        stage: str,
        algorithm: str,
    ) -> Path:
        return (
            Path(self.config.output_dir)
            / stage
            / algorithm
            / f"{slide_path.stem}{self.result_saver().extension}"
        )

    def attach_output_collector(self, segmenter):
        """
        In cohort mode we override the algorithm-specific collector output directory.

        The algorithm config still decides WHAT to collect:
            collectors:
              - name: ThumbnailCollector
              - name: SegmentationOverlayCollector
              - name: HistogramCollector

        But cohort pipeline decides WHERE those files are saved:
            outputs/<stage>/<algorithm>/visualizations/
        """

        if hasattr(segmenter.config, "build_output_collector"):
            segmenter.output_collector = segmenter.config.build_output_collector(
                out_dir=self.visualization_dir()
            )

        return segmenter

    def collect_slides(self) -> list[Path]:
        input_dir = Path(self.config.input_dir)
        return sorted(input_dir.glob(self.config.pattern))

    def sort_slides(self, slide_paths: list[Path]) -> list[Path]:
        return sorted(
            slide_paths,
            key=lambda p: p.stat().st_size,
            reverse=True,
        )

    @abstractmethod
    def run_one(self, slide_path: Path):
        ...

    @abstractmethod
    def output_exists(self, slide_path: Path) -> bool:
        ...

    def filter_slides(self, slide_paths: list[Path]) -> list[Path]:
        overwrite = getattr(self.config, "overwrite", False)

        if overwrite:
            return slide_paths

        return [
            path
            for path in slide_paths
            if not self.output_exists(path)
        ]

    def run(self, slide_paths: list[Path] | None = None):
        if slide_paths is None:
            slide_paths = self.collect_slides()

        slide_paths = self.filter_slides(slide_paths)
        slide_paths = self.sort_slides(slide_paths)

        logger = CohortLogger(Path(self.config.output_dir) / "logs")

        parallel_workers = getattr(self.config, "parallel_workers", 1)

        if parallel_workers == 1:
            self._run_sequential(slide_paths, logger)
        else:
            self._run_parallel(slide_paths, logger, parallel_workers)

    def _run_sequential(self, slide_paths: list[Path], logger: CohortLogger):
        for path in slide_paths:
            basename = path.stem
            t0 = time.perf_counter()

            try:
                self.run_one(path)

                logger.log_processed(
                    self.stage_name,
                    path,
                    basename,
                    time.perf_counter() - t0,
                )

            except Exception as e:
                logger.log_error(
                    self.stage_name,
                    path,
                    basename,
                    e,
                )

    def _run_parallel(
        self,
        slide_paths: list[Path],
        logger: CohortLogger,
        workers: int,
    ):
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.run_one, path): path
                for path in slide_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                basename = path.stem
                t0 = time.perf_counter()

                try:
                    future.result()

                    logger.log_processed(
                        self.stage_name,
                        path,
                        basename,
                        time.perf_counter() - t0,
                    )

                except Exception as e:
                    logger.log_error(
                        self.stage_name,
                        path,
                        basename,
                        e,
                    )