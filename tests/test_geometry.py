import cv2
import numpy as np
import pytest
from lunar_registration.geometry.homography import estimate_homography


def test_estimate_homography_synthetic():
    # 4 known points forming a square
    pts_src = [cv2.KeyPoint(x=float(x), y=float(y), size=1.0) for x, y in [(0, 0), (100, 0), (100, 100), (0, 100)]]
    # Translated and scaled
    pts_ref = [cv2.KeyPoint(x=float(x * 1.5 + 20), y=float(y * 1.5 - 10), size=1.0) for x, y in [(0, 0), (100, 0), (100, 100), (0, 100)]]

    # Create dummy DMatch objects
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.1) for i in range(4)]

    H, mask = estimate_homography(pts_src, pts_ref, matches, reprojection_threshold=3.0)

    assert H.shape == (3, 3)
    assert len(mask) == 4
    assert mask.all()


def test_estimate_homography_insufficient_matches():
    pts = [cv2.KeyPoint(x=0.0, y=0.0, size=1.0)]
    matches = [cv2.DMatch(_queryIdx=0, _trainIdx=0, _distance=0.1)]

    with pytest.raises(ValueError):
        estimate_homography(pts, pts, matches)
