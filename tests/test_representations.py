"""
Unit tests for lunar terrain representations.
"""

import numpy as np
import pytest

from lunar_registration.preprocessing.representations import (
    get_representation,
    REPRESENTATION_REGISTRY,
)


@pytest.fixture
def sample_image():
    # Synthetic textured surface (128x128)
    rng = np.random.RandomState(42)
    img = rng.randint(40, 200, (128, 128), dtype=np.uint8)
    return img


@pytest.mark.parametrize("rep_name", [
    "raw",
    "clahe",
    "gradient",
    "local_contrast",
    "laplacian",
    "edges",
    "phase_congruency",
])
def test_all_representations_shape_and_dtype(sample_image, rep_name):
    rep = get_representation(sample_image, rep_name)
    assert isinstance(rep, np.ndarray)
    assert rep.shape == sample_image.shape
    assert rep.dtype == np.uint8
    assert rep.min() >= 0
    assert rep.max() <= 255


def test_invalid_representation(sample_image):
    with pytest.raises(ValueError, match="Unknown representation"):
        get_representation(sample_image, "non_existent_representation")


def test_representation_registry_completeness():
    expected = {"raw", "clahe", "gradient", "local_contrast", "laplacian", "edges", "phase_congruency"}
    assert expected.issubset(set(REPRESENTATION_REGISTRY.keys()))
