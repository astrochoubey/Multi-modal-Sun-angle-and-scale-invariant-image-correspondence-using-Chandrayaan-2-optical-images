import numpy as np
from lunar_registration.matching.matcher import match_descriptors


def test_match_descriptors_exact():
    # Synthetic identical descriptors
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
