"""
Mutual nearest-neighbor cross-check correspondence verification.
"""

from typing import List
import cv2
import numpy as np


def match_with_cross_check(
    desc1: np.ndarray,
    desc2: np.ndarray,
    norm_type: int = cv2.NORM_L2
) -> List[cv2.DMatch]:
    """
    Perform bi-directional nearest neighbor matching with cross-check.
    """
    matcher = cv2.BFMatcher(norm_type, crossCheck=True)
    return matcher.match(desc1, desc2)
