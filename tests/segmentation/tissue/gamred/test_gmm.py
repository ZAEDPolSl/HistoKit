import numpy as np
import pytest

from histokit.segmentation.tissue.gamred.gmm import (
    get_pixel_distribution,
    norm_pdf,
    EM_iter_hist,
    gmm_init_dp_hist,
    GaMRed_hist,
)


def test_get_pixel_distribution_shape():
    img = np.zeros((10, 10, 3), dtype=np.uint8)

    R, G, B = get_pixel_distribution(img)

    assert R.shape == (256,)
    assert G.shape == (256,)
    assert B.shape == (256,)


def test_get_pixel_distribution_counts_pixels():
    img = np.zeros((2, 2, 3), dtype=np.uint8)

    img[..., 0] = 10
    img[..., 1] = 20
    img[..., 2] = 30

    R, G, B = get_pixel_distribution(img)

    assert R[10] == 4
    assert G[20] == 4
    assert B[30] == 4


def test_get_pixel_distribution_removes_white_pixels():
    img = np.full((5, 5, 3), 255, dtype=np.uint8)

    R, G, B = get_pixel_distribution(img)

    assert R[255] == 0
    assert G[255] == 0
    assert B[255] == 0


def test_norm_pdf_shape():
    x = np.array([0, 1, 2])

    y = norm_pdf(x, mu=0, sigma=1)

    assert y.shape == (1, 3)


def test_norm_pdf_positive_values():
    x = np.array([-1, 0, 1])

    y = norm_pdf(x, mu=0, sigma=1)

    assert np.all(y > 0)


def test_gmm_init_dp_hist_returns_valid_parameters():
    x = np.arange(256)
    y = np.ones(256)

    alpha, mu, sigma = gmm_init_dp_hist(x, y, K=2)

    assert len(alpha) == 2
    assert len(mu) == 2
    assert len(sigma) == 2

    np.testing.assert_allclose(alpha.sum(), 1.0, atol=1e-6)

    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))


def test_em_iter_hist_returns_valid_parameters():
    x = np.arange(256)
    y = np.ones(256)

    alpha = np.array([0.5, 0.5])
    mu = np.array([50.0, 200.0])
    sigma = np.array([10.0, 10.0])

    alpha_est, mu_est, sigma_est, logL = EM_iter_hist(
        x,
        y,
        alpha,
        mu,
        sigma,
        SW=5,
    )

    assert len(alpha_est) == 2
    assert len(mu_est) == 2
    assert len(sigma_est) == 2

    np.testing.assert_allclose(
        alpha_est.sum(),
        1.0,
        atol=1e-6,
    )

    assert np.all(np.isfinite(mu_est))
    assert np.all(np.isfinite(sigma_est))
    assert np.isfinite(logL)

def test_gamred_hist_two_components():
    x = np.arange(256)

    y = (
        500 * np.exp(-(x - 60) ** 2 / (2 * 10 ** 2))
        + 500 * np.exp(-(x - 180) ** 2 / (2 * 10 ** 2))
    )

    y = np.round(y).astype(int)

    thr, bic, stats = GaMRed_hist(
        x,
        y,
        K=2,
        draw=False,
        SW=5,
    )

    assert np.isfinite(thr)
    assert np.isfinite(bic)

    assert stats["K"] == 2
    assert len(stats["alpha"]) == 2
    assert len(stats["mu"]) == 2
    assert len(stats["sigma"]) == 2


def test_gamred_hist_returns_sorted_means():
    x = np.arange(256)

    y = (
        500 * np.exp(-(x - 60) ** 2 / (2 * 10 ** 2))
        + 500 * np.exp(-(x - 180) ** 2 / (2 * 10 ** 2))
    )

    y = np.round(y).astype(int)

    _, _, stats = GaMRed_hist(
        x,
        y,
        K=2,
        draw=False,
        SW=5,
    )

    assert np.all(np.diff(stats["mu"]) >= 0)