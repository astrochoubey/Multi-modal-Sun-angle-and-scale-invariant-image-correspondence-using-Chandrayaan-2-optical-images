"""
Evaluation metrics, visualization, and benchmark harness.
"""

from lunar_registration.evaluation.metrics import calculate_rmse, calculate_inlier_ratio
from lunar_registration.evaluation.visualization import (
    draw_matches,
    draw_registration_checkerboard,
    draw_registration_blend,
    draw_grid_distribution_overlay,
)
from lunar_registration.evaluation.benchmark import RegistrationBenchmark
from lunar_registration.evaluation.confidence import compute_registration_confidence
from lunar_registration.evaluation.representation_benchmark import benchmark_representations_on_pair
from lunar_registration.evaluation.synthetic_suite import (
    generate_synthetic_lunar_terrain,
    create_ground_truth_transform,
    evaluate_against_ground_truth,
    run_synthetic_benchmark,
)

__all__ = [
    "calculate_rmse",
    "calculate_inlier_ratio",
    "draw_matches",
    "draw_registration_checkerboard",
    "draw_registration_blend",
    "draw_grid_distribution_overlay",
    "RegistrationBenchmark",
    "compute_registration_confidence",
    "benchmark_representations_on_pair",
    "generate_synthetic_lunar_terrain",
    "create_ground_truth_transform",
    "evaluate_against_ground_truth",
    "run_synthetic_benchmark",
]
