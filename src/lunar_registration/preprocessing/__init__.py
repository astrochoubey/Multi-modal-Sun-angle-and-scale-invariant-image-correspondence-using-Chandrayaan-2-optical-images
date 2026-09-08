"""
Preprocessing subpackage for lunar image conditioning and representation extraction.
"""

from lunar_registration.preprocessing.preprocessing import to_grayscale, apply_clahe, preprocess
from lunar_registration.preprocessing.normalization import percentile_normalize, min_max_normalize
from lunar_registration.preprocessing.illumination import apply_wall_filter, remove_low_frequency_shadows
from lunar_registration.preprocessing.enhancement import enhance_contrast, unsharp_mask
from lunar_registration.preprocessing.representations import (
    get_representation,
    RepresentationType,
    REPRESENTATION_REGISTRY,
    extract_raw_grayscale,
    extract_clahe,
    extract_gradient_magnitude,
    extract_local_contrast,
    extract_laplacian,
    extract_edges,
    extract_phase_congruency,
)

__all__ = [
    "to_grayscale",
    "apply_clahe",
    "preprocess",
    "percentile_normalize",
    "min_max_normalize",
    "apply_wall_filter",
    "remove_low_frequency_shadows",
    "enhance_contrast",
    "unsharp_mask",
    "get_representation",
    "RepresentationType",
    "REPRESENTATION_REGISTRY",
    "extract_raw_grayscale",
    "extract_clahe",
    "extract_gradient_magnitude",
    "extract_local_contrast",
    "extract_laplacian",
    "extract_edges",
    "extract_phase_congruency",
]
