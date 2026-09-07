import cv2
import numpy as np


def calculate_rmse(
    source_keypoints,
    reference_keypoints,
    matches,
    homography: np.ndarray,
) -> float:

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

    projected_points = cv2.perspectiveTransform(
        source_points,
        homography,
    )

    errors = (
        reference_points - projected_points
    ).reshape(-1, 2)

    squared_errors = np.sum(
        errors ** 2,
        axis=1,
    )

    return float(
        np.sqrt(np.mean(squared_errors))
    )


def calculate_inlier_ratio(
    inlier_mask: np.ndarray,
) -> float:

    if len(inlier_mask) == 0:
        return 0.0

    return float(np.mean(inlier_mask))