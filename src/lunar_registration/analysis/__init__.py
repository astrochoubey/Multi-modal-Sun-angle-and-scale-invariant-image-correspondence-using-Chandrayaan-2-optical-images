"""
Image and pair analysis subpackage.
"""

from lunar_registration.analysis.pair_characterization import (
    characterize_image_pair,
    compute_texture_strength,
    compute_illumination_difference,
    compute_spectral_scale_proxy,
    compute_terrain_relief_proxy,
)

__all__ = [
    "characterize_image_pair",
    "compute_texture_strength",
    "compute_illumination_difference",
    "compute_spectral_scale_proxy",
    "compute_terrain_relief_proxy",
]
