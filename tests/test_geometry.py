import cv2
import numpy as np
import pytest

from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.evaluation.metrics import calculate_rmse, calculate_inlier_ratio


def test_estimate_homography_and_rmse():
    # Generate synthetic 2D points under known affine transformation
    np.random.seed(123)
    pts_src = np.random.uniform(50, 450, (25, 2)).astype(np.float32)

    # Known transformation: 2x2 rotation/scale + translation
    angle = np.radians(10)
    c, s = np.cos(angle), np.sin(angle)
    H_true = np.array([
        [c, -s, 20.0],
        [s,  c, 15.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    pts_ref = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H_true).reshape(-1, 2)

    # Create dummy KeyPoint and DMatch objects
    kp_src = [cv2.KeyPoint(x=p[0], y=p[1], size=1.0) for p in pts_src]
    kp_ref = [cv2.KeyPoint(x=p[0], y=p[1], size=1.0) for p in pts_ref]
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(len(pts_src))]

    H_est, inliers = estimate_homography(kp_src, kp_ref, matches, reprojection_threshold=3.0)

    assert H_est is not None
    assert inliers.sum() >= 20

    inlier_ratio = calculate_inlier_ratio(inliers)
    assert inlier_ratio > 0.8

    rmse = calculate_rmse(kp_src, kp_ref, matches, H_est)
    assert rmse < 1.0  # sub-pixel error on synthetic noise-free points
