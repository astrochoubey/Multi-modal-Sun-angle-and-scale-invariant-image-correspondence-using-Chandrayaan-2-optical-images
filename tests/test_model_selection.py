"""
Unit tests for geometric model selection framework.
"""

import numpy as np
import pytest

from lunar_registration.geometry.model_selection import (
    fit_translation,
    fit_similarity,
    fit_affine,
    fit_homography_model,
    compare_and_select_model,
)


@pytest.fixture
def correspondence_points():
    rng = np.random.RandomState(42)
    # 20 points in 2D
    src = rng.uniform(20.0, 200.0, size=(20, 2)).astype(np.float32)
    return src


def test_fit_translation(correspondence_points):
    src = correspondence_points
    tx, ty = 12.0, -8.0
    ref = src + np.array([tx, ty], dtype=np.float32)

    H, inliers, rmse = fit_translation(src, ref, threshold=1.0)
    assert H.shape == (3, 3)
    assert inliers.sum() == len(src)
    assert rmse < 1e-4
    assert np.isclose(H[0, 2], tx, atol=1e-3)
    assert np.isclose(H[1, 2], ty, atol=1e-3)


def test_fit_similarity(correspondence_points):
    src = correspondence_points
    theta = np.radians(15.0)
    scale = 1.05
    R = np.array([
        [scale * np.cos(theta), -scale * np.sin(theta)],
        [scale * np.sin(theta),  scale * np.cos(theta)],
    ], dtype=np.float32)
    ref = (src @ R.T) + np.array([10.0, -5.0], dtype=np.float32)

    H, inliers, rmse = fit_similarity(src, ref, threshold=1.0)
    assert H.shape == (3, 3)
    assert inliers.sum() == len(src)
    assert rmse < 0.1


def test_fit_affine(correspondence_points):
    src = correspondence_points
    M = np.array([
        [1.02, 0.05, 15.0],
        [-0.03, 0.98, -10.0]
    ], dtype=np.float32)
    ref = (src @ M[:, :2].T) + M[:, 2]

    H, inliers, rmse = fit_affine(src, ref, threshold=1.0)
    assert H.shape == (3, 3)
    assert inliers.sum() == len(src)
    assert rmse < 0.1


def test_compare_and_select_model_parsimony(correspondence_points):
    src = correspondence_points
    # Pure translation data: parsimony should select 'translation'
    ref = src + np.array([5.0, 5.0], dtype=np.float32)

    result = compare_and_select_model(src, ref, threshold=1.0)
    assert result["selected_model"] == "translation"
    assert result["inliers"] == len(src)
    assert result["rmse"] < 0.05
    assert "translation" in result["all_models"]
    assert "homography" in result["all_models"]
