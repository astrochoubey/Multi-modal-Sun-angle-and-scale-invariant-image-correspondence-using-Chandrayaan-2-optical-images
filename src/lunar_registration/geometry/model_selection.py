"""
Geometric Model Selection Framework.
Evaluates and compares transformation models of increasing complexity:
- Translation (2 DOF)
- Similarity (4 DOF)
- Affine (6 DOF)
- Homography (8 DOF)

Governing Principle:
"Use the simplest transformation model that adequately explains the observed correspondences."
"""

from typing import Dict, Any, Tuple, List
import cv2
import numpy as np


def fit_translation(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit 2-DOF translation [tx, ty]."""
    diffs = ref_pts - src_pts
    tx = float(np.median(diffs[:, 0]))
    ty = float(np.median(diffs[:, 1]))

    H = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    proj = src_pts + np.array([tx, ty])
    residuals = np.linalg.norm(ref_pts - proj, axis=1)
    inliers = residuals <= threshold
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if np.sum(inliers) > 0 else 999.0

    return H, inliers, rmse


def fit_similarity(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit 4-DOF Similarity (translation + uniform scale + rotation)."""
    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=threshold
    )
    if M is None or inliers is None:
        return np.eye(3, dtype=np.float32), np.zeros(len(src_pts), dtype=bool), 999.0

    inliers_mask = inliers.ravel().astype(bool)
    H = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float32)

    proj = (np.dot(src_pts, M[:, :2].T) + M[:, 2]).reshape(-1, 2)
    residuals = np.linalg.norm(ref_pts - proj, axis=1)
    rmse = float(np.sqrt(np.mean(residuals[inliers_mask] ** 2))) if np.sum(inliers_mask) > 0 else 999.0

    return H, inliers_mask, rmse


def fit_affine(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit 6-DOF Affine (translation + scale + rotation + shear)."""
    M, inliers = cv2.estimateAffine2D(
        src_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=threshold
    )
    if M is None or inliers is None:
        return np.eye(3, dtype=np.float32), np.zeros(len(src_pts), dtype=bool), 999.0

    inliers_mask = inliers.ravel().astype(bool)
    H = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float32)

    proj = (np.dot(src_pts, M[:, :2].T) + M[:, 2]).reshape(-1, 2)
    residuals = np.linalg.norm(ref_pts - proj, axis=1)
    rmse = float(np.sqrt(np.mean(residuals[inliers_mask] ** 2))) if np.sum(inliers_mask) > 0 else 999.0

    return H, inliers_mask, rmse


def fit_homography_model(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit 8-DOF Homography (planar perspective projection)."""
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    H, inliers = cv2.findHomography(src_pts, ref_pts, method, threshold)
    if H is None or inliers is None:
        return np.eye(3, dtype=np.float32), np.zeros(len(src_pts), dtype=bool), 999.0

    inliers_mask = inliers.ravel().astype(bool)
    pts_reshaped = src_pts.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(pts_reshaped, H).reshape(-1, 2)
    residuals = np.linalg.norm(ref_pts - proj, axis=1)
    rmse = float(np.sqrt(np.mean(residuals[inliers_mask] ** 2))) if np.sum(inliers_mask) > 0 else 999.0

    return H, inliers_mask, rmse


def compare_and_select_model(
    source_points: np.ndarray,
    reference_points: np.ndarray,
    threshold: float = 3.0,
    improvement_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Fit all 4 models and select the most parsimonious model that adequately explains the data.
    """
    n_pts = len(source_points)
    if n_pts < 4:
        raise ValueError("At least 4 correspondence points required for model comparison.")

    models = {}

    # 1. Translation
    H_trans, inl_trans, rmse_trans = fit_translation(source_points, reference_points, threshold)
    models["translation"] = {
        "dof": 2,
        "H": H_trans,
        "inliers": int(np.sum(inl_trans)),
        "inlier_ratio": float(np.mean(inl_trans)),
        "rmse": float(rmse_trans),
        "mask": inl_trans,
    }

    # 2. Similarity
    H_sim, inl_sim, rmse_sim = fit_similarity(source_points, reference_points, threshold)
    models["similarity"] = {
        "dof": 4,
        "H": H_sim,
        "inliers": int(np.sum(inl_sim)),
        "inlier_ratio": float(np.mean(inl_sim)),
        "rmse": float(rmse_sim),
        "mask": inl_sim,
    }

    # 3. Affine
    H_aff, inl_aff, rmse_aff = fit_affine(source_points, reference_points, threshold)
    models["affine"] = {
        "dof": 6,
        "H": H_aff,
        "inliers": int(np.sum(inl_aff)),
        "inlier_ratio": float(np.mean(inl_aff)),
        "rmse": float(rmse_aff),
        "mask": inl_aff,
    }

    # 4. Homography
    H_homo, inl_homo, rmse_homo = fit_homography_model(source_points, reference_points, threshold)
    models["homography"] = {
        "dof": 8,
        "H": H_homo,
        "inliers": int(np.sum(inl_homo)),
        "inlier_ratio": float(np.mean(inl_homo)),
        "rmse": float(rmse_homo),
        "mask": inl_homo,
    }

    # Parsimony Selection Order: translation -> similarity -> affine -> homography
    hierarchy = ["translation", "similarity", "affine", "homography"]
    selected_name = "homography"  # Default fallback

    current_best = hierarchy[0]
    for candidate in hierarchy[1:]:
        curr_rmse = models[current_best]["rmse"]
        cand_rmse = models[candidate]["rmse"]
        curr_inl = models[current_best]["inliers"]
        cand_inl = models[candidate]["inliers"]

        # Meaningful improvement: cand_rmse decreases by >15% and inliers don't drop drastically
        if cand_inl >= curr_inl and (curr_rmse - cand_rmse) / (curr_rmse + 1e-4) >= improvement_threshold:
            current_best = candidate
        elif cand_inl > curr_inl * 1.25 and cand_rmse <= curr_rmse * 1.1:
            current_best = candidate

    selected_name = current_best
    best_model = models[selected_name]

    return {
        "selected_model": selected_name,
        "H": best_model["H"],
        "inliers": best_model["inliers"],
        "inlier_ratio": best_model["inlier_ratio"],
        "rmse": best_model["rmse"],
        "mask": best_model["mask"],
        "all_models": {
            k: {
                "dof": v["dof"],
                "inliers": v["inliers"],
                "inlier_ratio": round(v["inlier_ratio"], 3),
                "rmse": round(v["rmse"], 3),
            }
            for k, v in models.items()
        }
    }
