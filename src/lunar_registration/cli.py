"""
Command Line Interface for Lunar Image Registration.
Provides subcommands:
- register:    Classical baseline registration (SIFT + RANSAC homography)
- adaptive:    Full adaptive registration with pair characterization, representations,
               model selection, piecewise warping, and confidence scoring
- characterize: Quantitative image pair diagnostics and strategy selection
- benchmark:   Comparative representation benchmark on a given image pair
- synthetic:   Controlled ground-truth benchmark suite with simulated crater terrain
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.preprocessing import preprocess, to_grayscale
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.registration.register import warp_image
from lunar_registration.registration.adaptive_register import adaptive_register_images
from lunar_registration.analysis.pair_characterization import characterize_image_pair
from lunar_registration.adaptive.strategy_selector import select_adaptive_strategy
from lunar_registration.evaluation.representation_benchmark import benchmark_representations_on_pair
from lunar_registration.evaluation.synthetic_suite import run_synthetic_benchmark
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
    calculate_spatial_coverage,
)
from lunar_registration.evaluation.visualization import draw_matches


def load_config_dict(config_path: str | Path | None) -> dict:
    """Load configuration from YAML file if provided."""
    defaults = {
        "features": {"sift": {"n_features": 5000}},
        "matching": {"ratio_threshold": 0.75},
        "geometry": {"reprojection_threshold_px": 5.0},
    }
    if not config_path:
        return defaults

    path = Path(config_path)
    if not path.exists():
        return defaults

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def run_register_command(
    source_path: str,
    reference_path: str,
    output_dir_str: str = "outputs",
    config_path: Optional[str] = None,
    **kwargs,
) -> dict:
    """Execute classical baseline registration."""
    print("=" * 60)
    print("Classical Baseline Registration (SIFT + RANSAC Homography)")
    print("=" * 60)

    cfg = load_config_dict(config_path)
    n_features = cfg.get("features", {}).get("sift", {}).get("n_features", 5000)
    ratio_threshold = cfg.get("matching", {}).get("ratio_threshold", 0.75)
    reproj_thresh = cfg.get("geometry", {}).get("reprojection_threshold_px", 5.0)

    print("Loading images...")
    source = load_image(source_path)
    reference = load_image(reference_path)

    print("Preprocessing (Standard CLAHE)...")
    source_processed = preprocess(source)
    reference_processed = preprocess(reference)

    print(f"Detecting SIFT features (max {n_features})...")
    source_keypoints, source_descriptors = detect_and_compute(source_processed, n_features=n_features)
    reference_keypoints, reference_descriptors = detect_and_compute(reference_processed, n_features=n_features)

    print(f"  Source keypoints:    {len(source_keypoints)}")
    print(f"  Reference keypoints: {len(reference_keypoints)}")

    if len(source_keypoints) < 4 or len(reference_keypoints) < 4:
        print("\n[ERROR] Insufficient keypoints extracted for registration.")
        raise RuntimeError("Insufficient keypoints extracted for registration.")

    print(f"Matching descriptors with Lowe's ratio test (threshold={ratio_threshold})...")
    matches = match_descriptors(source_descriptors, reference_descriptors, ratio_threshold=ratio_threshold)
    print(f"  Good matches:        {len(matches)}")

    if len(matches) < 4:
        print("\n[ERROR] Not enough matches for homography estimation (< 4).")
        raise RuntimeError("Not enough matches for homography estimation (< 4).")

    print(f"Estimating 8-DOF Homography with RANSAC (reproj_threshold={reproj_thresh})...")
    homography, inlier_mask = estimate_homography(
        source_keypoints,
        reference_keypoints,
        matches,
        reprojection_threshold=reproj_thresh,
    )

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = calculate_inlier_ratio(inlier_mask)

    print(f"  Inliers:             {inlier_count}")
    print(f"  Inlier ratio:        {inlier_ratio:.3f}")

    print("Warping source image...")
    registered = warp_image(source, homography, reference.shape)

    inlier_matches = [
        match
        for match, is_inlier in zip(matches, inlier_mask)
        if is_inlier
    ]

    rmse = calculate_rmse(
        source_keypoints,
        reference_keypoints,
        inlier_matches,
        homography,
    )
    spatial_coverage = calculate_spatial_coverage(
        reference_keypoints,
        inlier_matches,
        reference.shape,
    )
    print(f"  Reprojection RMSE:   {rmse:.4f} pixels")
    print(f"  Spatial Coverage:    {spatial_coverage:.2%}")

    output_dir = Path(output_dir_str)
    matches_dir = output_dir / "matches"
    registered_dir = output_dir / "registered"
    metrics_dir = output_dir / "metrics"

    matches_dir.mkdir(parents=True, exist_ok=True)
    registered_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    match_visualization = draw_matches(
        source,
        source_keypoints,
        reference,
        reference_keypoints,
        matches,
        inlier_mask,
    )

    cv2.imwrite(str(matches_dir / "matches.png"), match_visualization)
    cv2.imwrite(str(matches_dir / "correspondences.png"), match_visualization)
    cv2.imwrite(str(registered_dir / "registered.png"), registered)
    cv2.imwrite(str(registered_dir / "registered_image.tif"), registered)

    metrics = {
        "keypoints_source": len(source_keypoints),
        "keypoints_reference": len(reference_keypoints),
        "source_keypoints": len(source_keypoints),
        "reference_keypoints": len(reference_keypoints),
        "raw_matches": len(matches),
        "good_matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "rmse_pixels": rmse,
        "spatial_coverage": spatial_coverage,
        "homography": homography.tolist(),
    }

    with open(metrics_dir / "results.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nRegistration complete.")
    print(f"  Registered image:     {registered_dir / 'registered.png'}")
    print(f"  Match visualization:  {matches_dir / 'matches.png'}")
    print(f"  Metrics:              {metrics_dir / 'results.json'}")

    return metrics


# Alias for backwards compatibility
register_images = run_register_command


def run_adaptive_command(args):
    """Execute full adaptive multi-modal registration pipeline."""
    print("=" * 65)
    print("Adaptive Multi-Modal Lunar Image Registration (SIH 26166)")
    print("=" * 65)

    source_path = Path(args.source)
    reference_path = Path(args.reference)
    out_dir = Path(args.output_dir)

    result = adaptive_register_images(
        source_path=source_path,
        reference_path=reference_path,
        output_dir=out_dir,
        gsd_source=args.gsd_source,
        gsd_reference=args.gsd_reference,
        force_model=args.force_model,
        enable_piecewise=not args.no_piecewise,
        apply_spatial_filter=args.spatial_filter,
    )

    diag = result["diagnostics"]
    print("\n1. Image Pair Diagnostics:")
    print(f"  - Texture Energy:       {diag['texture_energy_ratio']:.3f} (Entropy Src: {diag['entropy_source']:.2f}, Ref: {diag['entropy_reference']:.2f})")
    print(f"  - Solar Azimuth Delta:  {diag['solar_azimuth_delta_deg']:.1f}°")
    print(f"  - Relief Shadow Proxy:  {diag['relief_shadow_proxy']:.3f}")
    print(f"  - Dynamic Range Ratio:  {diag['dynamic_range_ratio']:.3f}")
    print(f"  - GSD Scale Ratio:      {diag['estimated_gsd_scale_ratio']:.2f}x")

    strat = result["strategy"]
    print("\n2. Adaptive Strategy Selected:")
    print(f"  - Representation:       {strat['representation']}")
    print(f"  - Candidate Models:     {strat['candidate_models']}")
    print(f"  - Piecewise Warp:       {strat['enable_piecewise']}")
    print(f"  - Subpixel Refine:      {strat['enable_subpixel']}")
    print(f"  - Spatial Regularize:   {strat['spatial_regularization']}")

    reg = result["registration"]
    print("\n3. Registration Convergence:")
    print(f"  - Inliers:              {reg['inliers']} / {reg['raw_matches']} ({reg['inlier_ratio']*100:.1f}%)")
    print(f"  - Selected Model:       {reg['selected_model']}")
    print(f"  - Inlier Reproj RMSE:   {reg['inlier_rmse_pixels']:.3f} px")
    print(f"  - Global Check RMSE:    {reg['global_check_rmse_pixels']:.3f} px")

    subpx = result["subpixel"]
    print("\n4. Sub-Pixel Precision Refinement:")
    print(f"  - Enabled:              {subpx['enabled']}")
    if subpx["enabled"]:
        print(f"  - Mean Subpixel Offset: {subpx['mean_subpixel_offset_px']:.3f} px")
        print(f"  - Refined Match Points: {subpx['refined_count']}")

    conf = result["confidence"]
    print("\n5. Quality Assurance & Trustworthiness Score:")
    print(f"  - Score:                {conf['confidence_score']:.4f} / 1.0000")
    print(f"  - Classification:       {conf['category']}")
    print(f"  - Trustworthy:          {conf['trustworthy']}")
    print(f"  - Rationale:            {conf['rationale']}")

    spatial = result["spatial_distribution"]
    print("\n6. Spatial Match Regularization:")
    print(f"  - Grid Coverage Ratio:  {spatial['spatial_coverage']*100:.1f}% of cells occupied")
    print(f"  - Gini Concentration:   {spatial['gini_concentration_coefficient']:.3f} (0=uniform, 1=clustered)")

    print("\n" + "=" * 65)
    print(f"Outputs written to: {out_dir.resolve()}")
    print(f"  - Registered Image:     {out_dir / 'registered.png'}")
    print(f"  - Match Correspondences:{out_dir / 'matches.png'}")
    print(f"  - Seam Checkerboard:    {out_dir / 'checkerboard.png'}")
    print(f"  - False Color Overlay:  {out_dir / 'blend.png'}")
    print(f"  - Spatial Grid Plot:    {out_dir / 'spatial_distribution.png'}")
    print(f"  - Comprehensive Report: {out_dir / 'registration_report.json'}")
    print("=" * 65)


def run_characterize_command(args):
    """Execute image pair characterization."""
    print("=" * 60)
    print("Lunar Image Pair Characterization")
    print("=" * 60)

    img1 = load_image(args.source)
    img2 = load_image(args.reference)

    diag = characterize_image_pair(img1, img2, gsd1=args.gsd_source, gsd2=args.gsd_reference)
    strategy = select_adaptive_strategy(diag)

    print(json.dumps(diag, indent=2))
    print("\nRecommended Strategy:")
    print(json.dumps(strategy.to_dict(), indent=2))

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump({"diagnostics": diag, "recommended_strategy": strategy.to_dict()}, f, indent=4)
        print(f"\nSaved diagnostics to {out_p}")


def run_benchmark_command(args):
    """Run representation benchmark on a given pair."""
    print("=" * 60)
    print("Representation Benchmark across Conditioned Filters")
    print("=" * 60)

    img1 = load_image(args.source)
    img2 = load_image(args.reference)

    out_p = Path(args.output) if args.output else Path("outputs/benchmark/benchmark_results.json")
    summary = benchmark_representations_on_pair(img1, img2, output_dir=out_p.parent, pair_id="cli_pair")
    results = summary["results"]

    print(f"\n{'Representation':<22} {'Matches':<9} {'Inliers':<9} {'Inlier Ratio':<14} {'RMSE (px)':<10}")
    print("-" * 65)
    for m in results:
        rep = m["representation"]
        if not m.get("converged", False) and m.get("inliers", 0) < 4:
            print(f"{rep:<22} {m['matches']:<9} {m['inliers']:<9} {'FAILED':<14} {'N/A':<10}")
        else:
            print(f"{rep:<22} {m['matches']:<9} {m['inliers']:<9} {m['inlier_ratio']:<14.3f} {m['rmse_pixels']:<10.3f}")
    print("-" * 65)
    print(f"Best Representation: {summary['best_representation']}")
    print(f"Benchmark summary saved.")


def run_synthetic_command(args):
    """Run synthetic benchmark suite."""
    print("=" * 60)
    print("Synthetic Lunar Ground-Truth Evaluation Suite")
    print("=" * 60)

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs/synthetic")
    results = run_synthetic_benchmark(output_dir=out_dir, image_size=(args.size, args.size))

    print(f"\n{'Test Case':<25} {'Transform':<14} {'Raw Corner Err':<16} {'CLAHE Corner Err':<18} {'Gradient Corner Err':<20}")
    print("-" * 95)
    for test_name, data in results.items():
        t_type = data["ground_truth_transform_type"]
        reps = data["representations"]
        raw_err = reps.get("raw", {}).get("mean_corner_error_px", 999.0)
        clahe_err = reps.get("clahe", {}).get("mean_corner_error_px", 999.0)
        grad_err = reps.get("gradient", {}).get("mean_corner_error_px", 999.0)

        raw_str = f"{raw_err:.2f} px" if raw_err < 900 else "FAIL"
        clahe_str = f"{clahe_err:.2f} px" if clahe_err < 900 else "FAIL"
        grad_str = f"{grad_err:.2f} px" if grad_err < 900 else "FAIL"

        print(f"{test_name:<25} {t_type:<14} {raw_str:<16} {clahe_str:<18} {grad_str:<20}")
    print("-" * 95)
    print(f"Detailed results saved to: {out_dir / 'synthetic_benchmark_results.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Lunar Image Registration Toolkit (SIH 26166)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # 1. Classical Baseline Register
    register_parser = subparsers.add_parser(
        "register",
        help="Run classical baseline SIFT + RANSAC homography registration",
    )
    register_parser.add_argument("--source", required=True, help="Path to source image")
    register_parser.add_argument("--reference", required=True, help="Path to reference image")
    register_parser.add_argument("--config", default=None, help="Path to YAML configuration file")
    register_parser.add_argument("--output-dir", default="outputs", help="Directory for output files")

    # 2. Adaptive Register
    adaptive_parser = subparsers.add_parser(
        "adaptive",
        help="Run adaptive registration pipeline with model selection and confidence scoring",
    )
    adaptive_parser.add_argument("--source", required=True, help="Path to source image")
    adaptive_parser.add_argument("--reference", required=True, help="Path to reference image")
    adaptive_parser.add_argument("--gsd-source", type=float, default=None, help="GSD of source in m/px")
    adaptive_parser.add_argument("--gsd-reference", type=float, default=None, help="GSD of reference in m/px")
    adaptive_parser.add_argument("--output-dir", default="outputs/adaptive", help="Output directory")
    adaptive_parser.add_argument("--force-model", choices=["translation", "similarity", "affine", "homography"], default=None, help="Force specific geometric model")
    adaptive_parser.add_argument("--no-piecewise", action="store_true", help="Disable piecewise local refinement")
    adaptive_parser.add_argument("--spatial-filter", action="store_true", help="Apply spatial match bucketing")

    # 3. Pair Characterize
    char_parser = subparsers.add_parser(
        "characterize",
        help="Diagnose image pair properties (texture, illumination, relief proxy, scale)",
    )
    char_parser.add_argument("--source", required=True, help="Path to source image")
    char_parser.add_argument("--reference", required=True, help="Path to reference image")
    char_parser.add_argument("--gsd-source", type=float, default=None, help="GSD of source in m/px")
    char_parser.add_argument("--gsd-reference", type=float, default=None, help="GSD of reference in m/px")
    char_parser.add_argument("--output", default=None, help="Optional output JSON path")

    # 4. Benchmark Representations
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark feature extraction across representation methods on an image pair",
    )
    bench_parser.add_argument("--source", required=True, help="Path to source image")
    bench_parser.add_argument("--reference", required=True, help="Path to reference image")
    bench_parser.add_argument("--output", default=None, help="Output JSON path")

    # 5. Synthetic Ground Truth Benchmark
    synth_parser = subparsers.add_parser(
        "synthetic",
        help="Run controlled ground-truth benchmark suite on simulated lunar crater surfaces",
    )
    synth_parser.add_argument("--output-dir", default="outputs/synthetic", help="Output directory")
    synth_parser.add_argument("--size", type=int, default=512, help="Synthetic image dimension (NxN)")

    args = parser.parse_args()

    if args.command == "register":
        run_register_command(args.source, args.reference, output_dir_str=args.output_dir, config_path=args.config)
    elif args.command == "adaptive":
        run_adaptive_command(args)
    elif args.command == "characterize":
        run_characterize_command(args)
    elif args.command == "benchmark":
        run_benchmark_command(args)
    elif args.command == "synthetic":
        run_synthetic_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()