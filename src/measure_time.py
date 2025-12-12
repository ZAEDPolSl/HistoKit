import openslide
from src.histo_kit.patches_extraction.patches_extractor import PatchesExtractor
from src.histo_kit.utils.wsi import load_wsi_mag
import time

des_mag = 40
wsi_path = "/mnt/data/Datasets/Compass/HE/4-13_he_2.svs"

start = time.time()
wsi = openslide.OpenSlide(wsi_path)
region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, des_mag, allow_upscaling=True)
extractor = PatchesExtractor([region],"wsi1", out_dir="test_extract_wsi", patch_size=256, overlap=0)
extractor.extract_patches()
end = time.time()
print(f"Patch Extraction Time without mask: {end - start} seconds")