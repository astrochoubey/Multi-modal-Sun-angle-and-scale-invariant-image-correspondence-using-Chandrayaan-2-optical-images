"""
Robust consensus estimators (RANSAC, USAC_MAGSAC, LMEDS).
"""

from typing import Tuple, List
import cv2
import numpy as np


def fit_homography_robust(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    method: str = "USAC_MAGSAC",
    threshold: float = 3.0,
    max_iters: int = 10000,
    confidence: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit homography with selectable consensus estimator.
    """
    method_map = {
        "RANSAC": cv2.RANSAC,
        "USAC_MAGSAC": cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC,
        "LMEDS": cv2.LMEDS,
    }
    cv_method = method_map.get(method.upper(), cv2.RANSAC)

    H, mask = cv2.findHomography(
        src_pts,
        ref_pts,
        method=cv_method,
        ransacReprojThreshold=threshold,
        maxIters=max_iters,
        confidence=confidence,
    )
    if H is None or mask is None:
        raise RuntimeError("Geometric consensus estimation failed.")

    return H, mask.ravel().astype(bool)
