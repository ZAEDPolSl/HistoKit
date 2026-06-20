from .base import ExtractionStrategy
from ..datasets.grid import GridExtractorDataset


class GridExtractionStrategy(ExtractionStrategy):
    def __init__(self, overlap=0.7,
                 grid_offset=0.5):

        self.overlap = overlap
        self.grid_offset = grid_offset

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
        return GridExtractorDataset(
            region=region_np,
            patch_size=patch_size,
            overlap=self.overlap,
            pad_value=pad_value,
            grid_offset=self.grid_offset,
            prep_fn=prep_fn,
            exclude_fn=exclude_fn,
            aug_fn=aug_fn,
            bbox_list=bbox_list,
            patch_writer=patch_writer,
        )
