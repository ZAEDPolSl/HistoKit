from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from histokit.segmentation.config import BaseAlgorithmConfig
from ...savers.base import Saver
from ..logger import CohortLogger
from tqdm import tqdm

class BaseCohortPipeline(ABC):
    """Base class for cohort-level processing pipelines.

    This class implements common logic for running a processing stage over a
    collection of whole-slide images. It handles slide collection, output path
    construction, skipping already processed slides, sequential or parallel
    execution, and logging of processed slides and errors.

    Subclasses are expected to define the processing logic for a single slide
    by implementing :meth:`run_one` and :meth:`output_exists`.

    Attributes
    ----------
    stage_name : str
        Name of the processing stage, used in logs and progress bars.
    result_subdir : str
        Name of the output subdirectory for this pipeline stage.
    result_dir_name : str or None, default=None
        Optional subdirectory inside the stage directory where result files are
        saved. If ``None``, results are saved directly in the stage directory.
    config : BaseSegmentationConfig
        Pipeline configuration object.
    """

    stage_name: str
    result_subdir: str

    def __init__(self, config: BaseAlgorithmConfig):
        """Initialize the cohort pipeline.

        Parameters
        ----------
        config : BaseAlgorithmConfig
            Pipeline configuration. See `BaseAlgorithmConfig` documentation.
        """
        self.config = config

    def result_saver(self):
        """Return saver used for writing pipeline results.

        Returns
        -------
        Saver
            Saver instance selected from ``config.saver``. If the configuration
            does not define a saver, ``"hdf5"`` is used by default.
        """
        return Saver(getattr(self.config, "saver", "hdf5"))

    def stage_dir(self, algorithm: str | None = None) -> Path:
        """Return stage output directory.

        Parameters
        ----------
        algorithm : str, optional
            Algorithm name. If omitted, ``config.algorithm`` is used.

        Returns
        -------
        pathlib.Path
            Path to the algorithm-specific stage directory.
        """
        algorithm = algorithm or self.config.algorithm

        return (
            Path(self.config.output_dir)
            / self.result_subdir
            / algorithm
        )
    
    def result_dir(self, algorithm: str | None = None) -> Path:
        """Return directory where result files are saved.

        Parameters
        ----------
        algorithm : str, optional
            Algorithm name. If omitted, ``config.algorithm`` is used.

        Returns
        -------
        pathlib.Path
            Path to the result directory.
        """
        stage_dir = self.stage_dir(algorithm=algorithm)

        if self.result_dir_name is None:
            return stage_dir

        return stage_dir / self.result_dir_name

    def visualization_dir(self, algorithm: str | None = None) -> Path:
        """Return directory where visualization files are saved.

        Parameters
        ----------
        algorithm : str, optional
            Algorithm name. If omitted, ``config.algorithm`` is used.

        Returns
        -------
        pathlib.Path
            Path to the visualization directory.
        """
        return self.stage_dir(algorithm=algorithm) / "visualizations"

    def output_path(self, slide_path: Path, algorithm: str | None = None) -> Path:
        """Return output file path for a slide.

        Parameters
        ----------
        slide_path : pathlib.Path
            Path to the input slide.
        algorithm : str, optional
            Algorithm name. If omitted, ``config.algorithm`` is used.

        Returns
        -------
        pathlib.Path
            Output path for the slide result.
        """
        return (
            self.result_dir(algorithm=algorithm)
            / f"{slide_path.stem}{self.result_saver().extension}"
        )

    def result_path(self, mask_dir: Path, slide_path: Path) -> Path:
        return Path(mask_dir) / f"{slide_path.stem}{self.result_saver().extension}"

    def attach_output_collector(self, step_runner) -> object:

        if hasattr(step_runner.config, "build_output_collector"):
            step_runner.output_collector = step_runner.config.build_output_collector(
                out_dir=self.visualization_dir())

        return step_runner

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
        for path in tqdm(
            slide_paths,
            total=len(slide_paths),
            desc=self.stage_name,
            unit="slide",
        ):
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

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=self.stage_name,
                unit="slide",
            ):
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