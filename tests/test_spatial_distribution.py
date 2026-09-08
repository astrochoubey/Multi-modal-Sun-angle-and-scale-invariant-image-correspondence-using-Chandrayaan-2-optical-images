"""
Unit tests for spatial match distribution and regularization.
"""

import numpy as np
import pytest
import cv2

from lunar_registration.matching.spatial_distribution import (
    analyze_spatial_distribution,
    filter_matches_spatially,
)


def test_analyze_spatial_distribution_uniform():
    shape = (100, 100)
    # Generate points evenly in every cell of a 4x4 grid
    pts = []
    for r in range(4):
        for c in range(4):
            pts.append([c * 25 + 12, r * 25 + 12])
    pts = np.array(pts, dtype=np.float32)

    diag = analyze_spatial_distribution(pts, shape, grid_size=(4, 4))
    assert diag["total_cells"] == 16
    assert diag["occupied_cells"] == 16
    assert diag["spatial_coverage"] == 1.0
    # Perfect equality has low Gini coefficient
    assert diag["gini_concentration_coefficient"] < 0.1


def test_analyze_spatial_distribution_clustered():
    shape = (100, 100)
    # All points clustered inside a single cell (0, 0)
    pts = np.array([[5, 5], [8, 10], [12, 14], [15, 8]], dtype=np.float32)

    diag = analyze_spatial_distribution(pts, shape, grid_size=(4, 4))
    assert diag["occupied_cells"] == 1
    assert diag["spatial_coverage"] == 1.0 / 16.0
    # Clustered points have high Gini coefficient
    assert diag["gini_concentration_coefficient"] > 0.8


def test_filter_matches_spatially():
    shape = (100, 100)
    # 50 matches all concentrated in top-left cell
    kp_src = [cv2.KeyPoint(x=float(i), y=float(i), size=1.0) for i in range(50)]
    kp_ref = [cv2.KeyPoint(x=5.0, y=5.0, size=1.0) for _ in range(50)]
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=float(i)) for i in range(50)]

    filtered = filter_matches_spatially(
        kp_src,
        kp_ref,
        matches,
        reference_shape=shape,
        grid_size=(4, 4),
        max_matches_per_cell=10,
    )
    assert len(filtered) == 10
    # The 10 filtered should be the lowest distance matches
    distances = [m.distance for m in filtered]
    assert max(distances) < 10.0
