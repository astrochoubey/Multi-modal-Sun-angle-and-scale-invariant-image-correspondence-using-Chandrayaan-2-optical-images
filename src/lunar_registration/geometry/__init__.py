"""
Geometry estimation, consensus, model selection, and piecewise registration subpackage.
"""

from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.geometry.ransac import fit_homography_robust
from lunar_registration.geometry.transforms import project_points, compute_reprojection_residuals
from lunar_registration.geometry.refinement import subpixel_dft_registration
from lunar_registration.geometry.model_selection import compare_and_select_model
from lunar_registration.geometry.piecewise import PiecewiseRegistrar

__all__ = [
    "estimate_homography",
    "fit_homography_robust",
    "project_points",
    "compute_reprojection_residuals",
    "subpixel_dft_registration",
    "compare_and_select_model",
    "PiecewiseRegistrar",
]
