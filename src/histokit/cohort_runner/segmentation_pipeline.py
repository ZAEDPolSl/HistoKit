from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..savers.base import BaseSaver
from ..slide.slide import Slide
from .config import PipelineConfig


class TissueSegPipeline:

    def __init__(self, config: PipelineConfig, saver: BaseSaver, visualizations=None):
        self.segmenter = self._create_segmenter(config)
        self.saver = saver
        self.visualizations = visualizations

    def run(self, slide_path, save_path = None):
        slide = Slide(slide_path)

        result, mask = self.segmenter.segment(slide)

        if save_path is not None:
            self.saver.save(save_path, result)

        return mask

    def run_parallel(
            self,
            slide_paths,
            max_workers: int = 4,
            chunk_multiplier: int = 4,
            fail_fast: bool = False,
    ):
        if not slide_paths:
            return []

        num_chunks = min(max_workers * chunk_multiplier, len(slide_paths))

        slide_chunks = self._chunkify(slide_paths, num_chunks)

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            future_to_chunk = {
                executor.submit(self._process_chunk, chunk): chunk
                for chunk in slide_chunks
            }

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]

                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)

                except Exception as e:
                    print(f"[ERROR] Chunk failed {chunk}: {e}")

                    if fail_fast:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

        return results

    def _process_chunk(self, slide_paths):
        results = []
        for slide_path in slide_paths:
            result = self.run(slide_path)
            results.append((slide_path, result))
        return results

    @staticmethod
    def _chunkify(items, num_chunks):
        if num_chunks <= 0:
            return []

        k, m = divmod(len(items), num_chunks)
        return [
            items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(num_chunks)
        ]

    @staticmethod
    def _create_segmenter(config: PipelineConfig):
        try:
            segmenter_cls = SEGMENTERS[config.method]
        except KeyError:
            raise ValueError(
                f"Unknown segmentation method: {config.method}"
            )

        return segmenter_cls(config)

    @staticmethod
    def _create_saver(saver_type: str, **kwargs):
        try:
            saver_cls = SAVERS[saver_type]
        except KeyError:
            raise ValueError(f"Unknown saver type: {saver_type}")
        return saver_cls(**kwargs)

