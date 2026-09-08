"""
Representation Benchmark Engine
Evaluates multiple terrain-structure representations on the exact same image pair.
Answers the scientific question:
"Which representation gives the most stable correspondences under illumination variation?"
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import json

import cv2
import numpy as np

from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.representations import (
    get_representation,
    REPRESENTATION_REGISTRY,
)
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.evaluation.metrics import calculate_rmse, calculate_inlier_ratio


def evaluate_single_representation(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    representation_name: str,
    n_features: int = 5000,
    ratio_threshold: float = 0.75,
    reproj_threshold: float = 3.0,
    grid_size: tuple[int, int] = (8, 8),
) -> Dict[str, Any]:
    """
    Run the registration pipeline using a specific representation.
    """
    t0 = time.perf_counter()

    # 1. Transform images to representation
    src_rep = get_representation(source_image, representation_name)
    ref_rep = get_representation(reference_image, representation_name)

    # 2. Extract features
    src_kp, src_desc = detect_and_compute(src_rep, n_features=n_features)
    ref_kp, ref_desc = detect_and_compute(ref_rep, n_features=n_features)

    num_src_kp = len(src_kp)
    num_ref_kp = len(ref_kp)

    if src_desc is None or ref_desc is None or num_src_kp < 4 or num_ref_kp < 4:
        runtime = time.perf_counter() - t0
        return {
            "representation": representation_name,
            "keypoints_source": num_src_kp,
            "keypoints_reference": num_ref_kp,
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse_pixels": 0.0,
            "spatial_coverage": 0.0,
            "converged": False,
            "runtime_seconds": round(runtime, 4),
            "failure_reason": "insufficient_features",
        }

    # 3. Match
    matches = match_descriptors(src_desc, ref_desc, ratio_threshold=ratio_threshold)
    num_matches = len(matches)

    if num_matches < 4:
        runtime = time.perf_counter() - t0
        return {
            "representation": representation_name,
            "keypoints_source": num_src_kp,
            "keypoints_reference": num_ref_kp,
            "matches": num_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse_pixels": 0.0,
            "spatial_coverage": 0.0,
            "converged": False,
            "runtime_seconds": round(runtime, 4),
            "failure_reason": "insufficient_matches",
        }

    # 4. Robust Geometry
    try:
        H, inlier_mask = estimate_homography(
            src_kp, ref_kp, matches, reprojection_threshold=reproj_threshold
        )
        inliers_count = int(inlier_mask.sum())
        inlier_ratio = calculate_inlier_ratio(inlier_mask)

        inlier_matches = [m for m, is_inl in zip(matches, inlier_mask) if is_inl]
        rmse = calculate_rmse(src_kp, ref_kp, inlier_matches, H) if inliers_count >= 4 else 0.0

        # Spatial Coverage across grid
        h_ref, w_ref = reference_image.shape[:2]
        gw, gh = w_ref / grid_size[0], h_ref / grid_size[1]
        occupied = set()
        for m in inlier_matches:
            pt = ref_kp[m.trainIdx].pt
            cx = min(int(pt[0] / gw), grid_size[0] - 1)
            cy = min(int(pt[1] / gh), grid_size[1] - 1)
            occupied.add((cx, cy))
        coverage = len(occupied) / float(grid_size[0] * grid_size[1])

        converged = bool(inliers_count >= 15 and rmse <= reproj_threshold and coverage >= 0.20)
    except Exception as e:
        inliers_count = 0
        inlier_ratio = 0.0
        rmse = 0.0
        coverage = 0.0
        converged = False

    runtime = time.perf_counter() - t0

    return {
        "representation": representation_name,
        "keypoints_source": num_src_kp,
        "keypoints_reference": num_ref_kp,
        "matches": num_matches,
        "inliers": inliers_count,
        "inlier_ratio": round(float(inlier_ratio), 4),
        "rmse_pixels": round(float(rmse), 4),
        "spatial_coverage": round(float(coverage), 4),
        "converged": converged,
        "runtime_seconds": round(runtime, 4),
    }


def run_representation_benchmark(
    source: str | Path | np.ndarray,
    reference: str | Path | np.ndarray,
    representations: Optional[List[str]] = None,
    output_dir: Optional[str | Path] = "outputs/metrics",
    pair_id: str = "custom_pair",
) -> Dict[str, Any]:
    """
    Benchmark an image pair across multiple representations.
    """
    if isinstance(source, (str, Path)):
        src_img = load_image(source)
    else:
        src_img = source

    if isinstance(reference, (str, Path)):
        ref_img = load_image(reference)
    else:
        ref_img = reference

    target_reps = representations or ["raw", "clahe", "gradient", "local_contrast", "laplacian", "edges", "phase_congruency"]

    print(f"\n" + "=" * 80)
    print(f"REPRESENTATION BENCHMARK: Pair '{pair_id}'")
    print(f"Testing {len(target_reps)} representations: {target_reps}")
    print("=" * 80)

    results = []
    for rep in target_reps:
        print(f"  --> Evaluating representation: {rep:<18}", end="", flush=True)
        res = evaluate_single_representation(src_img, ref_img, rep)
        results.append(res)
        status = "CONVERGED" if res.get("converged") else "UNCONVERGED"
        print(f"[{status}] Inliers: {res['inliers']:<4} Ratio: {res['inlier_ratio']:.2f} RMSE: {res['rmse_pixels']:.2f}px Cov: {res['spatial_coverage']:.2f} ({res['runtime_seconds']:.2f}s)")

    # Find the best representation by harmonic score: inlier_ratio * coverage / (rmse + 1e-3)
    scored = []
    for r in results:
        if r.get("converged") and r.get("rmse_pixels", 0) > 0:
            score = (r["inlier_ratio"] * r["spatial_coverage"]) / (r["rmse_pixels"] + 0.1)
        else:
            score = r["inlier_ratio"] * 0.1
        scored.append((score, r["representation"]))

    scored.sort(reverse=True)
    best_rep = scored[0][1] if scored else "clahe"

    benchmark_summary = {
        "pair_id": pair_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_representation": best_rep,
        "representations_evaluated": target_reps,
        "results": results,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        json_file = out_path / f"representation_benchmark_{pair_id.lower()}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=4)
    return benchmark_summary


# Alias for backward compatibility and clean API
benchmark_representations_on_pair = run_representation_benchmark
