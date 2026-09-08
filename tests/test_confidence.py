"""
Unit tests for explainable registration confidence scoring.
"""

import numpy as np
import pytest

from lunar_registration.evaluation.confidence import compute_registration_confidence


def test_high_confidence_registration():
    res = compute_registration_confidence(
        inliers_count=80,
        inlier_ratio=0.75,
        reprojection_rmse=0.8,
        spatial_coverage=0.55,
        residuals=np.array([0.7, 0.8, 0.9, 0.8]),
    )
    assert res["confidence_score"] >= 0.75
    assert res["category"] == "HIGH_CONFIDENCE"
    assert res["trustworthy"] is True
    assert 0.0 <= res["confidence_score"] <= 1.0
    assert "component_scores" in res
    assert "rationale" in res


def test_rejected_registration_due_to_low_inliers():
    res = compute_registration_confidence(
        inliers_count=4,
        inlier_ratio=0.1,
        reprojection_rmse=6.5,
        spatial_coverage=0.05,
    )
    assert res["category"] == "REJECTED"
    assert res["trustworthy"] is False
    assert res["confidence_score"] < 0.30


def test_boundedness():
    # Extreme inputs should stay bounded in [0.0, 1.0]
    res_high = compute_registration_confidence(
        inliers_count=10000,
        inlier_ratio=1.0,
        reprojection_rmse=0.0,
        spatial_coverage=1.0,
    )
    assert res_high["confidence_score"] == 1.0

    res_low = compute_registration_confidence(
        inliers_count=0,
        inlier_ratio=0.0,
        reprojection_rmse=100.0,
        spatial_coverage=0.0,
    )
    assert res_low["confidence_score"] == 0.0
