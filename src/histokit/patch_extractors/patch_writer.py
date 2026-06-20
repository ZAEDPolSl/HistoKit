from pathlib import Path
from PIL import Image


class PatchImageWriter:
    def __init__(self, out_dir, prefix="patch"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def __call__(self, patch, x_start, y_start, x_end, y_end, exclude=False):
        if exclude:
            return  # Skip writing excluded patches
        filename = f"{self.prefix}_{x_start}_{y_start}_{x_end}_{y_end}.png"
        print(f"Writing {filename}")
        Image.fromarray(patch).save(self.out_dir / filename)