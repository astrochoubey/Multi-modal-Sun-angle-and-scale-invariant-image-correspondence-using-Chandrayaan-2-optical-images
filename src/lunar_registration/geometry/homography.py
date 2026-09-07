import cv2
import numpy as np


def estimate_homography(
    source_keypoints,
    reference_keypoints,
    matches,
    reprojection_threshold: float = 5.0,
):
    """
    Estimate a homography using RANSAC.

    Returns
    -------
    H : np.ndarray
        3x3 homography matrix.

    mask : np.ndarray
        RANSAC inlier mask.
    """

    if len(matches) < 4:
        raise ValueError(
            "At least 4 matches are required to estimate a homography."
        )

    source_points = np.float32(
        [
            source_keypoints[m.queryIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    reference_points = np.float32(
        [
            reference_keypoints[m.trainIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        reprojection_threshold,
    )

    if H is None or mask is None:
        raise RuntimeError(
            "Homography estimation failed."
        )

    return H, mask.ravel().astype(bool)