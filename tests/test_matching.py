import numpy as np
import cv2
from lunar_registration.matching.matcher import match_descriptors


def test_match_descriptors():
    # Create synthetic identical descriptors
    desc1 = np.random.randn(50, 128).astype(np.float32)
    desc2 = desc1.copy()

    # Normalize descriptors (as SIFT does)
    desc1 /= np.linalg.norm(desc1, axis=1, keepdims=True)
    desc2 /= np.linalg.norm(desc2, axis=1, keepdims=True)

    matches = match_descriptors(desc1, desc2, ratio_threshold=0.75)

    assert len(matches) > 0
    # Every point should match to itself with near-zero distance
    for m in matches:
        assert m.queryIdx == m.trainIdx
        assert m.distance < 1e-4
