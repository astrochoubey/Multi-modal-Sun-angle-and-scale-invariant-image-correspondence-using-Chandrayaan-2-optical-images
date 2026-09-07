import numpy as np

from lunar_registration.preprocessing.preprocessing import (
    to_grayscale,
    apply_clahe,
)


def test_to_grayscale():

    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = to_grayscale(image)

    assert result.shape == (100, 100)


def test_clahe():

    image = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    result = apply_clahe(image)

    assert result.shape == image.shape