"""
Registration Confidence Scoring Engine.
Computes an explainable, bounded composite confidence score answering:
"How trustworthy is this registration?"

Mathematical Formulation:
S_conf = w_inl * s_inl + w_rat * s_rat + w_rmse * s_rmse + w_cov * s_cov + w_res * s_res

Where:
- s_inl   = min(1.0, N_inliers / 50.0)             (Weight: 0.25)
- s_rat   = min(1.0, inlier_ratio / 0.50)           (Weight: 0.25)
- s_rmse  = max(0.0, 1.0 - rmse / 3.0)              (Weight: 0.25)
- s_cov   = min(1.0, spatial_coverage / 0.40)       (Weight: 0.15)
- s_res   = max(0.0, 1.0 - residual_std / 2.0)      (Weight: 0.10)
"""

from typing import Dict, Any, List
import numpy as np


def compute_registration_confidence(
    inliers_count: int,
    inlier_ratio: float,
    reprojection_rmse: float,
    spatial_coverage: float,
    residuals: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Calculate an explainable registration confidence score in [0.0, 1.0].
    """
    # 1. Inlier count score (saturates at 50 points)
    s_inl = min(1.0, max(0.0, inliers_count / 50.0))

    # 2. Inlier ratio score (saturates at 50% inliers)
    s_rat = min(1.0, max(0.0, inlier_ratio / 0.50))

    # 3. RMSE score (1.0 at 0 px, 0.0 at >= 3.0 px)
    s_rmse = max(0.0, 1.0 - (reprojection_rmse / 3.0))

    # 4. Spatial coverage score (saturates at 40% grid coverage)
    s_cov = min(1.0, max(0.0, spatial_coverage / 0.40))

    # 5. Residual consistency (low standard deviation of residuals indicates uniform fit)
    if residuals is not None and len(residuals) > 0:
        res_std = float(np.std(residuals))
        s_res = max(0.0, 1.0 - (res_std / 2.0))
    else:
        res_std = 0.0
        s_res = s_rmse  # Fallback to rmse score

    # Weights
    w_inl = 0.25
    w_rat = 0.25
    w_rmse = 0.25
    w_cov = 0.15
    w_res = 0.10

    composite_score = (
        w_inl * s_inl +
        w_rat * s_rat +
        w_rmse * s_rmse +
        w_cov * s_cov +
        w_res * s_res
    )
    composite_score = round(float(np.clip(composite_score, 0.0, 1.0)), 4)

    # Categorization
    if inliers_count < 8 or reprojection_rmse > 5.0 or composite_score < 0.30:
        category = "REJECTED"
        trustworthy = False
    elif composite_score >= 0.75:
        category = "HIGH_CONFIDENCE"
        trustworthy = True
    elif composite_score >= 0.50:
        category = "MEDIUM_CONFIDENCE"
        trustworthy = True
    else:
        category = "LOW_CONFIDENCE"
        trustworthy = False

    return {
        "confidence_score": composite_score,
        "category": category,
        "trustworthy": trustworthy,
        "component_scores": {
            "inlier_count_score": round(s_inl, 3),
            "inlier_ratio_score": round(s_rat, 3),
            "rmse_score": round(s_rmse, 3),
            "spatial_coverage_score": round(s_cov, 3),
            "residual_consistency_score": round(s_res, 3),
        },
        "weights": {
            "inlier_count": w_inl,
            "inlier_ratio": w_rat,
            "rmse": w_rmse,
            "spatial_coverage": w_cov,
            "residual_consistency": w_res,
        },
        "rationale": (
            f"Confidence {category} ({composite_score:.2f}) evaluated from {inliers_count} inliers "
            f"({inlier_ratio*100:.1f}% ratio), {reprojection_rmse:.2f}px RMSE, and {spatial_coverage*100:.1f}% spatial coverage."
        ),
        "limitations": "Heuristic composite metric; assumes near-planar or well-sampled terrain correspondences."
    }
