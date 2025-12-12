import os

import pytest
from PIL import Image
import numpy as np
from src.histo_kit.grand_qc.visualisation import make_overlay


def test_gaussian_blurring(patch_path):
    patch = Image.open(patch_path).convert("RGB")

def test_motion_blurring(patch_path):
    patch = Image.open(patch_path).convert("RGB")

def test_median_blurring(patch_path):
    patch = Image.open(patch_path).convert("RGB")

def test_jpeg_compression(patch_path):
    patch = Image.open(patch_path).convert("RGB")

def test_gaussian_noise(patch_path):
    patch = Image.open(patch_path).convert("RGB")

def test_salt_and_pepper_noise(patch_path):
    patch = Image.open(patch_path).convert("RGB")

