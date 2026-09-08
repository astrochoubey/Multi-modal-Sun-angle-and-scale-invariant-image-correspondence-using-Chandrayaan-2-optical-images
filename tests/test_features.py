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
        # Fallback to high-contrast textured image for CI or clean checkout
        np.random.seed(0)
        image = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
        cv2.circle(image, (150, 150), 50, 255, -1)
        cv2.rectangle(image, (50, 50), (100, 100), 0, -1)

    keypoints, descriptors = detect_and_compute(image)

    assert len(keypoints) > 0
    assert descriptors is not None
    assert descriptors.shape[1] == 128