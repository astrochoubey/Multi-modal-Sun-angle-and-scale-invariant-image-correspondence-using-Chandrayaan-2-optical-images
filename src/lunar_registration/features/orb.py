"""
ORB (Oriented FAST and Rotated BRIEF) feature extractor.
"""

import cv2
import numpy as np


def detect_and_compute_orb(
    image: np.ndarray,
    n_features: int = 5000,
    scale_factor: float = 1.2,
    n_levels: int = 8,
):
    """
    Detect ORB keypoints and compute binary descriptors.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    n_features : int
        Maximum number of keypoints.

    Returns
    -------
    keypoints : list of cv2.KeyPoint
    descriptors : np.ndarray (uint8 binary descriptors)
    """
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
    )
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors
