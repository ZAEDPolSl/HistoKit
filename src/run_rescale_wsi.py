import glob
import os
import argparse
import numpy as np
from openslide import OpenSlide
from tifffile import tifffile
from tqdm import tqdm
from PIL import Image

from src.histo_kit.wsi_utils.patches import load_wsi_mag

"""
Script for loading wsi with desired resolution. If desired resolution is not available, image is rescaled from the 
"""

parser = argparse.ArgumentParser()
parser.add_argument('--wsi_dir', type=str, help='Input directory with WSIs', default='/mnt/data/Datasets/HE_data/Pietrus/Healthy/')
parser.add_argument("--out_folder", type=str, help="Input directory with masks created by grandQC for detected regions with file extension. ", default="healthy_mag2.5")
parser.add_argument('--desired_mag', help="Desired slide magnification.", default=2.5, type = float)



if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    slides = sorted(glob.glob(os.path.join(args.wsi_dir, '*.svs')))
    for s_path in tqdm(slides):
        print(s_path)
        try:
            slide = OpenSlide(s_path)
            s_path = os.path.join(os.path.join(args.wsi_dir, s_path))
            wsi = OpenSlide(s_path)
            region, scale_val, info, mpp_slide, ratio = load_wsi_mag(wsi, args.desired_mag, rescale_method=Image.LANCZOS, verbose=True, allow_upscaling=True)

            res_path = os.path.join(args.out_folder, os.path.basename(s_path).split('.')[0] + '.tiff')

            region = np.array(region).astype(np.uint8)
            tifffile.imwrite(res_path, region, photometric='rgb', compression='lzw')
        except Exception as e:
            print(e)