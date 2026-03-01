from pathlib import Path

import cv2
import numpy as np

from .base import VisualizationStep
from ...slide.slide import Slide

class Thumbnail(VisualizationStep):

    def __call__(self, data: dict, slide: Slide = None, out_dir: Path = None, sub_dir_name="raw_small", vis_mag: float = 0.625):
        save_dir = out_dir / sub_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        img = slide.read_region(mag=vis_mag)
        basename = data.get("basename")
        img.save(save_dir / f"{basename}.png")

class TissueSegmentation(VisualizationStep):

    def __call__(self, data: dict, slide: Slide = None, out_dir: Path = None, sub_dir_name="bg_removal_contour_vis", vis_mag: float = 0.625):
        save_dir = out_dir / sub_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        img = slide.read_region(mag=vis_mag)
        mask = data['mask']
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 5)
        img.save(save_dir / f"{data.name}.png")