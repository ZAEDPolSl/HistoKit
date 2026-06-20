from abc import ABC, abstractmethod

class ExtractionStrategy(ABC):

    @abstractmethod
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
        raise NotImplementedError