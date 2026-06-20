from pathlib import Path

import cv2
import numpy as np
import pytest

from histokit.stain_normalisation import MacenkoExtractor, VahadaneExtractor


TEST_PATH = Path(__file__).parent.parent / "data/stain_normalization"


def read_rgb(path):
	img = cv2.imread(str(path))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img.astype(np.uint8)


@pytest.fixture
def target_img():
	return read_rgb(TEST_PATH / "target.png")


@pytest.mark.parametrize("extractor_cls", [MacenkoExtractor, VahadaneExtractor])
def test_stain_extractor_returns_normalized_matrix(extractor_cls, target_img):
	extractor = extractor_cls()

	stain_matrix = extractor.get_stain_matrix(target_img)

	assert isinstance(stain_matrix, np.ndarray)
	assert stain_matrix.shape == (2, 3)
	assert np.isfinite(stain_matrix).all()
	np.testing.assert_allclose(np.linalg.norm(stain_matrix, axis=1), np.ones(2), atol=1e-6)


@pytest.mark.parametrize("extractor_cls", [MacenkoExtractor, VahadaneExtractor])
def test_stain_extractor_rejects_non_rgb_uint8_input(extractor_cls):
	extractor = extractor_cls()

	with pytest.raises(AssertionError, match="Image should be RGB uint8"):
		extractor.get_stain_matrix(np.zeros((8, 8), dtype=np.uint8))
