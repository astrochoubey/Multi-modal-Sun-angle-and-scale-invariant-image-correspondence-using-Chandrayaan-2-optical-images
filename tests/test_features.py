import cv2

from lunar_registration.features.sift import (
    detect_and_compute,
)


def test_sift():

    image = cv2.imread(
        "data/raw/reference.png",
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        import numpy as np
        image = np.zeros((256, 256), dtype=np.uint8)
        # Draw some synthetic lunar craters so SIFT detects keypoints
        cv2.circle(image, (128, 128), 30, 200, -1)
        cv2.circle(image, (135, 135), 25, 40, -1)
        cv2.circle(image, (64, 64), 15, 180, -1)
        cv2.circle(image, (200, 180), 20, 220, -1)

    keypoints, descriptors = detect_and_compute(
        image
    )

    assert len(keypoints) > 0
    assert descriptors is not None