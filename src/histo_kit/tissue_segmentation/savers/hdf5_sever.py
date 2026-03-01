import os
import h5py
from .base_server import BaseSaver
from ..pipeline.registry import register_saver

@register_saver("hdf5")
class HDF5Saver(BaseSaver):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, slide_file: str, result):
        basename = os.path.splitext(os.path.basename(slide_file))[0]
        out_path = os.path.join(self.output_dir, f"{basename}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("mask", data=result["mask"], compression="gzip")
            f.attrs["method"] = result.get("method", "unknown")
        print(f"[HDF5Saver] Saved result to {out_path}")