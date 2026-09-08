"""
Unit tests for lunar image pair characterization.
"""

import numpy as np
import pytest

from lunar_registration.analysis.pair_characterization import (
    compute_texture_strength,
    compute_illumination_difference,
    compute_spectral_scale_proxy,
    compute_terrain_relief_proxy,
    characterize_image_pair,
)


@pytest.fixture
def image_pair():
    rng = np.random.RandomState(42)
    img1 = rng.randint(50, 180, (128, 128), dtype=np.uint8)
    # img2 with shifted brightness and different texture
    img2 = np.clip(img1.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    return img1, img2


def test_compute_texture_strength(image_pair):
    img1, _ = image_pair
    tex = compute_texture_strength(img1)
    assert "level" in tex
    assert tex["level"] in ["texture_rich", "medium_texture", "weak_texture"]
    assert "spatial_variance" in tex
    assert "shannon_entropy" in tex
    assert "gradient_density" in tex
    assert tex["shannon_entropy"] >= 0.0


def test_compute_illumination_difference(image_pair):
    img1, img2 = image_pair
    illum = compute_illumination_difference(img1, img2)
    assert "category" in illum
    assert "bhattacharyya_distance" in illum
    assert 0.0 <= illum["bhattacharyya_distance"] <= 1.0
    assert "shadow_disparity" in illum


def test_compute_spectral_scale_proxy(image_pair):
    img1, img2 = image_pair
    scale_info = compute_spectral_scale_proxy(img1, img2)
    assert "spectral_high_frequency_ratio" in scale_info
    assert "is_significant_scale_jump" in scale_info
    assert scale_info["spectral_high_frequency_ratio"] > 0.0


def test_compute_terrain_relief_proxy(image_pair):
    img1, _ = image_pair
    relief = compute_terrain_relief_proxy(img1)
    assert "complexity_class" in relief
    assert "edge_density" in relief
    assert "laplacian_variance" in relief
    assert "disclaimer" in relief


def test_characterize_image_pair_full(image_pair):
    img1, img2 = image_pair
    diag = characterize_image_pair(img1, img2, gsd1=0.5, gsd2=5.0)
    assert "source_texture" in diag
    assert "reference_texture" in diag
    assert "illumination" in diag
    assert "scale" in diag
    assert "terrain_relief_proxy" in diag
    assert diag["scale"]["metadata_scale_ratio"] == 0.1
