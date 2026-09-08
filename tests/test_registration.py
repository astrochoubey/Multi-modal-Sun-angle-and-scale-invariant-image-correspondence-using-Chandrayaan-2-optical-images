import cv2
import numpy as np
from lunar_registration.registration.register import warp_image


def test_warp_image():
    # Identity homography with grayscale image
    H = np.eye(3, dtype=np.float32)
    source_gray = np.ones((100, 100), dtype=np.uint8) * 128
    warped_gray = warp_image(source_gray, H, reference_shape=(100, 100))

    assert warped_gray.shape == (100, 100)
    assert np.allclose(warped_gray, source_gray)

    # Identity homography with 3-channel color image
    source_color = np.ones((100, 100, 3), dtype=np.uint8) * 128
    warped_color = warp_image(source_color, H, (100, 100))

    assert warped_color.shape == (100, 100, 3)
    np.testing.assert_array_equal(warped_color, source_color)


def test_warp_image_translation():
    # Translate by 10 px right, 20 px down
    H = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 20.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    source = np.zeros((100, 100), dtype=np.uint8)
    source[40:60, 40:60] = 255

    warped = warp_image(source, H, reference_shape=(100, 100))

    assert warped.shape == (100, 100)
    # Target region should be translated to [60:80, 50:70]
    assert warped[60, 50] == 255
