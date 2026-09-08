"""
Lowe's ratio test correspondence filter.
"""

from typing import List
import cv2


def filter_ratio_test(
    knn_matches: List[List[cv2.DMatch]],
    ratio_threshold: float = 0.75
) -> List[cv2.DMatch]:
    """
    Apply Lowe's second-nearest-neighbor ratio test.
    """
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches
