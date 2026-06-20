import os
from pathlib import Path
import cv2
import pytest
import numpy as np
from matplotlib import pyplot as plt
from histokit.stain_normalisation.normalizers import StainingNormalizer

TEST_PATH = Path(__file__).parent.parent / "data/stain_normalisation/"

def read_rgb(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


@pytest.fixture
def target_img():
    return read_rgb(os.path.join(TEST_PATH, "target.png"))


@pytest.fixture
def img1():
    return read_rgb(os.path.join(TEST_PATH, "img1.png"))


@pytest.fixture
def img2():
    return read_rgb(os.path.join(TEST_PATH, "img2.png"))


@pytest.fixture
def img3():
    return read_rgb(os.path.join(TEST_PATH, "img3.png"))


@pytest.fixture
def img4():
    return read_rgb(os.path.join(TEST_PATH, "img4.png"))

@pytest.fixture
def images(target_img, img1, img2, img3, img4):
    return target_img, [img1, img2, img3, img4]


@pytest.mark.parametrize("method", ["macenko", "vahadane", "reinhard"])
def test_stain_normalizer_fit_returns_self(method, target_img):
    normalizer = StainingNormalizer(method)

    fitted = normalizer.fit(target_img)

    assert fitted is normalizer


@pytest.mark.parametrize("method", ["macenko", "vahadane", "reinhard"])
def test_stain_normalizer_transform_output_shape(method, images):
    target_img, source_imgs = images

    normalizer = StainingNormalizer(method)
    normalizer.fit(target_img)

    for img in source_imgs:
        normalized = normalizer.transform(img)

        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == img.shape


@pytest.mark.parametrize("method", ["macenko", "vahadane", "reinhard"])
def test_stain_normalizer_transform_output_dtype(method, images):
    target_img, source_imgs = images

    normalizer = StainingNormalizer(method)
    normalizer.fit(target_img)

    for img in source_imgs:
        normalized = normalizer.transform(img)

        assert normalized.dtype == np.uint8


@pytest.mark.parametrize("method", ["macenko", "vahadane", "reinhard"])
def test_stain_normalizer_transform_output_range(method, images):
    target_img, source_imgs = images

    normalizer = StainingNormalizer(method)
    normalizer.fit(target_img)

    for img in source_imgs:
        normalized = normalizer.transform(img)

        assert normalized.min() >= 0
        assert normalized.max() <= 255


def test_stain_normalizer_unknown_method_raises_error():
    with pytest.raises(ValueError, match="Unknown stain normalization method"):
        StainingNormalizer("unknown")


@pytest.mark.parametrize("method", ["MACENKO", "Vahadane", "REINHARD"])
def test_stain_normalizer_method_is_case_insensitive(method, target_img, img1):
    normalizer = StainingNormalizer(method)
    normalizer.fit(target_img)

    normalized = normalizer.transform(img1)

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == img1.shape
    assert normalized.dtype == np.uint8


@pytest.mark.parametrize("method", ["macenko", "vahadane", "reinhard"])
def test_stain_normalizer_does_not_modify_input(method, target_img, img1):
    target_copy = target_img.copy()
    img_copy = img1.copy()

    normalizer = StainingNormalizer(method)
    normalizer.fit(target_img)
    _ = normalizer.transform(img1)

    np.testing.assert_array_equal(target_img, target_copy)
    np.testing.assert_array_equal(img1, img_copy)


@pytest.mark.skip_ci
@pytest.mark.parametrize("method", ["macenko", "vahadane", "reinhard"])
def test_stain_normalizer_visualization(method, target_img, img1):

    normalizer = StainingNormalizer(method)

    normalizer.fit(target_img)

    normalized = normalizer.transform(img1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(target_img)
    axes[0].set_title("Target")
    axes[0].axis("off")

    axes[1].imshow(img1)
    axes[1].set_title("Original")
    axes[1].axis("off")

    axes[2].imshow(normalized)
    axes[2].set_title(f"Normalized ({method})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

@pytest.mark.skip_ci
def test_compare_stain_normalization_methods(target_img, img2):

    methods = ["macenko", "vahadane", "reinhard"]

    normalized_images = {}

    for method in methods:

        normalizer = StainingNormalizer(method)

        normalizer.fit(target_img)

        normalized_images[method] = normalizer.transform(img2)

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    # Target
    axes[0].imshow(target_img)
    axes[0].set_title("Target")
    axes[0].axis("off")

    # Original
    axes[1].imshow(img2)
    axes[1].set_title("Original")
    axes[1].axis("off")

    # Macenko
    axes[2].imshow(normalized_images["macenko"])
    axes[2].set_title("Macenko")
    axes[2].axis("off")

    # Vahadane
    axes[3].imshow(normalized_images["vahadane"])
    axes[3].set_title("Vahadane")
    axes[3].axis("off")

    # Reinhard
    axes[4].imshow(normalized_images["reinhard"])
    axes[4].set_title("Reinhard")
    axes[4].axis("off")

    plt.tight_layout()
    plt.show()