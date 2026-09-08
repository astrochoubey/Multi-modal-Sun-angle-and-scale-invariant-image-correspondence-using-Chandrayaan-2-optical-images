"""
Unit tests for piecewise / grid-based local registration.
"""

import numpy as np
import pytest

from lunar_registration.geometry.piecewise import PiecewiseRegistrar


@pytest.fixture
def synthetic_scene():
    h, w = 120, 120
    src_img = np.zeros((h, w), dtype=np.uint8)
    src_img[20:100, 20:100] = 180

    rng = np.random.RandomState(42)
    # 36 points scattered across the grid
    src_pts = rng.uniform(10.0, 110.0, size=(36, 2)).astype(np.float32)
    ref_pts = src_pts + np.array([4.0, -3.0], dtype=np.float32)
    global_H = np.array([
        [1.0, 0.0, 4.0],
        [0.0, 1.0, -3.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return src_img, src_pts, ref_pts, global_H, (h, w)


def test_piecewise_registrar_fit_and_warp(synthetic_scene):
    src_img, src_pts, ref_pts, global_H, shape = synthetic_scene
    registrar = PiecewiseRegistrar(grid_rows=2, grid_cols=2, min_cell_matches=3)

    fitted = registrar.fit(src_pts, ref_pts, shape, global_H)
    assert "tile_models" in fitted
    assert len(fitted["tile_models"]) == 4
    assert "cell_stats" in fitted

    warped = registrar.warp(src_img, shape, fitted)
    assert isinstance(warped, np.ndarray)
    assert warped.shape == shape
    assert warped.dtype == np.uint8
    assert warped.max() > 0
