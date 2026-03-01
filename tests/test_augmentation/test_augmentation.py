from PIL import Image
import os

from src.histo_kit.augmentation.base import Compose
from src.histo_kit.augmentation.blurring import JPEGCompression, MedianBlur, GaussianBlur, MotionBlur
from src.histo_kit.augmentation.color_augmentation import ColorJitter, SaltAndPepper, GaussianNoise
from src.histo_kit.augmentation.rotations import RandomFlip, RandomRotation


def test_aug(output_dir="/tmp/aug_test", n_samples=1):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load test image
    img = Image.open("/mnt/data/Tmp/jmerta/HE/tests/test_augmentation/image.png")

    # Define augmentation pipeline
    for i in range(10):
        transform = Compose([
            GaussianNoise(prob=1.0),
        ])
        aug_img = transform(img)
        save_path = os.path.join(f"GaussianNoise_{i}.png")
        aug_img.save(save_path)

    # Define augmentation pipeline
    transform = Compose([
        SaltAndPepper(prob=1.0),
    ])
    aug_img = transform(img)
    save_path = os.path.join(f"SaltAndPepper.png")
    aug_img.save(save_path)

    # Define augmentation pipeline
    transform = Compose([
        ColorJitter(prob=1.0),
    ])
    aug_img = transform(img)
    save_path = os.path.join(f"ColorJitter.png")
    aug_img.save(save_path)

    # Define augmentation pipeline
    transform = Compose([
        JPEGCompression(prob=1.0, quality_range=(5, 10))
    ])
    aug_img = transform(img)
    save_path = os.path.join(f"JPEGCompression.png")
    aug_img.save(save_path)

    transform = Compose([
        MedianBlur(prob=1.0),
    ])
    aug_img = transform(img)
    save_path = os.path.join(f"MedianBlur.png")
    aug_img.save(save_path)

    transform = Compose([
        GaussianBlur(prob=1.0),
    ])
    aug_img = transform(img)
    save_path = os.path.join(f"GaussianBlur.png")
    aug_img.save(save_path)

    transform = Compose([
        RandomFlip(prob=1.0),
    ])
    aug_img = transform(img)
    save_path = os.path.join(f"RandomFlip.png")
    aug_img.save(save_path)

    transform = Compose([
        MotionBlur(prob=1.0),
    ])
    aug_img = transform(img)
    save_path = os.path.join(f"MotionBlur.png")
    aug_img.save(save_path)



if __name__ == "__main__":
    test_aug()
