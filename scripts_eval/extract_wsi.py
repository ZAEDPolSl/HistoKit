import os

import numpy as np
import openslide
import pandas as pd
from tifffile import tifffile

from histokit.slide import Slide
from tqdm import tqdm

wsi_dir = "/mnt/warehouse/Projects/HE/Data/Artifacts Segmentation/TCGA_CompassNMD/svs/"
out_dir = "/mnt/warehouse/Projects/HE/Data/Artifacts Segmentation/TCGA_CompassNMD/10x/tissue_10x/"
os.makedirs(out_dir, exist_ok=True)

wsis = [os.path.join(wsi_dir, f) for f in os.listdir(wsi_dir) if f.endswith(".svs")]

paths_res = [os.path.basename(wsi.replace(".svs", ".tiff")) for wsi in tqdm(wsis)]

df = pd.DataFrame({"wsi": paths_res})
df["mpp"] = 1
df.to_csv(out_dir + "wsi_mpp.csv", index=False)

for wsi_path in tqdm(wsis):
    basename = os.path.splitext(os.path.basename(wsi_path))[0]

    slide = Slide(wsi_path)
    print(slide.properties)
    slide_10x = slide.read_region(mag=10)

    image = np.asarray(slide_10x)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    out_path = os.path.join(out_dir, basename + ".tiff")

    tifffile.imwrite(
        out_path,
        image,
        photometric="rgb",
        tile=(512, 512),
        compression=None,

        # 1 µm/px = 10 000 px/cm
        resolution=(10000, 10000),
        resolutionunit="CENTIMETER",

        # informacja zachowana w TIFF,
        # ale NIE jako openslide.objective-power
        description="MPP=1.0; aperio.AppMag=10",

        metadata=None,
    )

    s = openslide.OpenSlide(out_path)

    print("vendor:", s.properties.get("openslide.vendor"))
    print("levels:", s.level_count)
    print("dimensions:", s.level_dimensions)

    print("XResolution:", s.properties.get("tiff.XResolution"))
    print("YResolution:", s.properties.get("tiff.YResolution"))
    print("ResolutionUnit:", s.properties.get("tiff.ResolutionUnit"))

    print("MPP X:", s.properties.get("openslide.mpp-x"))
    print("MPP Y:", s.properties.get("openslide.mpp-y"))
    print("Objective:", s.properties.get("openslide.objective-power"))

