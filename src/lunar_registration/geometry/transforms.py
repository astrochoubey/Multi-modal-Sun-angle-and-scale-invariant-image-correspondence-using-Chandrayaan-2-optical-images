"""
Coordinate transformation and perspective projection mathematics.
"""

import cv2
import numpy as np


def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Project 2D points using a 3x3 homography matrix.
    """
    pts_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(pts_reshaped, H)
    return projected.reshape(-1, 2)


def compute_reprojection_residuals(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """Compute Euclidean distance residuals between projected and reference points."""
    proj = project_points(src_pts, H)
    return np.linalg.norm(proj - ref_pts, axis=1)
