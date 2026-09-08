"""
Generalized descriptor matching module.
"""

from typing import List
import cv2
import numpy as np


def match_descriptors_knn(
    desc1: np.ndarray,
    desc2: np.ndarray,
    k: int = 2,
    norm_type: int = cv2.NORM_L2
) -> List[List[cv2.DMatch]]:
    """Match descriptors using k-nearest neighbors."""
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    return matcher.knnMatch(desc1, desc2, k=k)
