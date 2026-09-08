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


def test_match_descriptors_exact():
    # Synthetic identical descriptors with distractors
    np.random.seed(42)
    desc1 = np.random.randn(20, 128).astype(np.float32)
    # Normalize descriptors like SIFT
    desc1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
    desc2 = desc1.copy()

    # Add dummy distractor descriptors
    distractors = np.random.randn(30, 128).astype(np.float32)
    distractors = distractors / np.linalg.norm(distractors, axis=1, keepdims=True)
    desc2_with_distractors = np.vstack([desc2, distractors])

    matches = match_descriptors(desc1, desc2_with_distractors, ratio_threshold=0.75)

    assert len(matches) > 0
    # Perfect matches should match queryIdx == trainIdx
    for m in matches:
        if m.trainIdx < 20:
            assert m.queryIdx == m.trainIdx
