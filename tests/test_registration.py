import numpy as np
from lunar_registration.registration.register import warp_image


def test_warp_image():
    source = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Identity homography
    H = np.eye(3, dtype=np.float32)

    registered = warp_image(source, H, (100, 100))

    assert registered.shape == (100, 100, 3)
    np.testing.assert_array_equal(registered, source)
