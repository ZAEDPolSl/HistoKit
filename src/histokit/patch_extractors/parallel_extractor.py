from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, default_collate
from ..slide import Slide
from .patch_writer import PatchImageWriter
from .strategies.grid import GridExtractionStrategy
from ..savers.hdf5 import HDF5Saver


class ParallelExtractor:
    def __init__(
        self,
        patch_size: int = 512,
        prep_fn=None,
        aug_fn=None,
        exclude_fn=None,
        mask_loader=HDF5Saver(),
        extraction_strategy=None,
        extract_mag: float = 5,
        pad_value: int = 255,
        save_patches: bool = False,
    ):
        self.patch_size = patch_size
        self.prep_fn = prep_fn
        self.aug_fn = aug_fn
        self.exclude_fn = exclude_fn
        self.mask_loader = mask_loader
        self.extract_mag = extract_mag
        self.pad_value = pad_value
        self.save_patches = save_patches

        self.extraction_strategy = extraction_strategy or GridExtractionStrategy()

    def extract_patches(
        self,
        slide_path,
        path_mask=None,
        out_dir=None,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        slide = Slide(slide_path)
        basename = Path(slide_path).stem

        if path_mask is not None:
            mask_data = self.mask_loader.load(path_mask)
        else:
            w, h = slide.get_size_at_mag(self.extract_mag)
            mask_data = {
                "mask": None,
                "bbox": [np.array([0, 0, w, h])],
                "mag_save": self.extract_mag
            }

        if self.save_patches:
            if out_dir is None:
                raise ValueError("out_dir must be provided when save_patches=True")

            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True)

        for obj_idx, (mask, bbox) in enumerate(
            zip(mask_data["mask"], mask_data["bbox"])
        ):

            mask[mask!=255] = 0
            region = slide.read_masked_object(
                bbox = bbox,
                mask = mask,
                mag_bbox=mask_data["mag_save"],
                mag=self.extract_mag,
                pad_value=self.pad_value,
            )

            region_np = np.asarray(region)

            patch_writer = None
            if self.save_patches:
                patch_writer = PatchImageWriter(
                    out_dir=out_dir,
                    prefix=f"{basename}_R{obj_idx}",
                )

            ds = self.extraction_strategy.build_dataset(
                region_np=region_np,
                bbox_list=[bbox],
                patch_size=self.patch_size,
                pad_value=self.pad_value,
                exclude_fn=self.exclude_fn,
                prep_fn=self.prep_fn,
                aug_fn=self.aug_fn,
                patch_writer=patch_writer,
            )

            dataloader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.collate_remove_none,
                pin_memory=pin_memory,
            )

            for batch in dataloader:
                if batch is None:
                    continue

                yield batch


    @staticmethod
    def collate_remove_none(batch):
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            return None

        return default_collate(batch)

