"""
Descriptor matching, correspondence filtering, and spatial distribution subpackage.
"""

from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.matching.descriptor_matching import match_descriptors_knn
from lunar_registration.matching.ratio_test import filter_ratio_test
from lunar_registration.matching.cross_check import match_with_cross_check
from lunar_registration.matching.spatial_distribution import (
    analyze_spatial_distribution,
    filter_matches_spatially,
)

__all__ = [
    "match_descriptors",
    "match_descriptors_knn",
    "filter_ratio_test",
    "match_with_cross_check",
    "analyze_spatial_distribution",
    "filter_matches_spatially",
]
