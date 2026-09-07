import cv2

from lunar_registration.features.sift import (
    detect_and_compute,
)


def test_sift():

    image = cv2.imread(
        "data/raw/reference.png",
        cv2.IMREAD_GRAYSCALE,
    )

    keypoints, descriptors = detect_and_compute(
        image
    )

    assert len(keypoints) > 0
    assert descriptors is not None