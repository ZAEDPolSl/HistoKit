from .base import ExtractionStrategy
from ..datasets.random import RandomExtractorDataset


class RandomExtractionStrategy(ExtractionStrategy):
    def __init__(self, patch_number=1000, seed=42):
        self.patch_number = patch_number
        self.seed = seed

    def build_dataset(
        self,
            region_np,
            bbox_list,
            patch_size,
            pad_value,
            exclude_fn,
            prep_fn,
            aug_fn,
            patch_writer,
    ):
        return RandomExtractorDataset(
            region = region_np,
            bbox_list = bbox_list,
            patch_size = patch_size,
            pad_value=pad_value,
            prep_fn=prep_fn,
            aug_fn=aug_fn,
            exclude_fn=exclude_fn,
            patch_writer=patch_writer,
            patch_number=self.patch_number,
            seed=self.seed,
        )
