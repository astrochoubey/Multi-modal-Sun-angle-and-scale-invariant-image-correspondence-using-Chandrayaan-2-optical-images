import cv2
import numpy as np


def match_descriptors(
    source_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    ratio_threshold: float = 0.75,
):
    """
    Match SIFT descriptors using KNN matching
    followed by Lowe's ratio test.
    """

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    raw_matches = matcher.knnMatch(
        source_descriptors,
        reference_descriptors,
        k=2,
    )

    good_matches = []

    for pair in raw_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return good_matches