from pathlib import Path
import cv2
import numpy as np

from lunar_registration.features.sift import (
    detect_and_compute,
)


def test_sift():
    img_path = Path("data/raw/reference.png")
    if img_path.exists():
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = None

    if image is None:
        # Fallback synthetic lunar craters so SIFT detects keypoints
        image = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(image, (128, 128), 30, 200, -1)
        cv2.circle(image, (135, 135), 25, 40, -1)
        cv2.circle(image, (64, 64), 15, 180, -1)
        cv2.circle(image, (200, 180), 20, 220, -1)

    keypoints, descriptors = detect_and_compute(image)

    assert len(keypoints) > 0
    assert descriptors is not None
    assert descriptors.shape[1] == 128