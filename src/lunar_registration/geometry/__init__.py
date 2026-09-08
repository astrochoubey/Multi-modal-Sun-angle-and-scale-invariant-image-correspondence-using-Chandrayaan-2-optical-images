"""
Geometry estimation, consensus, and sub-pixel refinement subpackage.
"""

from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.geometry.ransac import fit_homography_robust
from lunar_registration.geometry.transforms import project_points, compute_reprojection_residuals
from lunar_registration.geometry.refinement import subpixel_dft_registration

__all__ = [
    "estimate_homography",
    "fit_homography_robust",
    "project_points",
    "compute_reprojection_residuals",
    "subpixel_dft_registration",
]
