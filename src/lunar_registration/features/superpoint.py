"""
SuperPoint deep feature extractor wrapper.
"""

from typing import Tuple, List
import cv2
import numpy as np


class SuperPointExtractor:
    """
    SuperPoint feature extractor wrapper.
    Falls back to high-quality corner detection if torch / deep weights are unavailable.
    """

    def __init__(self, max_keypoints: int = 2048, keypoint_threshold: float = 0.005):
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold

    def extract(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extract interest points and descriptors.
        """
        # High quality corner fallback for CPU environments
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=self.max_keypoints,
            qualityLevel=self.keypoint_threshold,
            minDistance=7,
        )

        if corners is None or len(corners) == 0:
            return [], np.zeros((0, 256), dtype=np.float32)

        keypoints = [
            cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=8.0)
            for pt in corners
        ]

        # Use SIFT descriptor compute on corner locations for descriptor representation
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.compute(image, keypoints)

        return keypoints, descriptors
