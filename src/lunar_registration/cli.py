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
)
from lunar_registration.evaluation.visualization import draw_matches


def run_register_command(source_path: str, reference_path: str, output_dir_str: str = "outputs"):
    """Execute classical baseline registration."""
    print("=" * 60)
    print("Classical Baseline Registration (SIFT + RANSAC Homography)")
    print("=" * 60)

    print("Loading images...")
    source = load_image(source_path)
    reference = load_image(reference_path)

    print("Preprocessing (Standard CLAHE)...")
    source_processed = preprocess(source)
    reference_processed = preprocess(reference)

    print("Detecting SIFT features...")
    source_keypoints, source_descriptors = detect_and_compute(source_processed)
    reference_keypoints, reference_descriptors = detect_and_compute(reference_processed)

    print(f"  Source keypoints:    {len(source_keypoints)}")
    print(f"  Reference keypoints: {len(reference_keypoints)}")

    if len(source_keypoints) < 4 or len(reference_keypoints) < 4:
        print("\n[ERROR] Insufficient keypoints extracted for registration.")
        sys.exit(1)

    print("Matching descriptors with Lowe's ratio test (threshold=0.75)...")
    matches = match_descriptors(source_descriptors, reference_descriptors)
    print(f"  Good matches:        {len(matches)}")

    if len(matches) < 4:
        print("\n[ERROR] Not enough matches for homography estimation (< 4).")
        sys.exit(1)

    print("Estimating 8-DOF Homography with RANSAC...")
    homography, inlier_mask = estimate_homography(
        source_keypoints,
        reference_keypoints,
        matches,
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
    print(f"  Reprojection RMSE:   {rmse:.4f} pixels")

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
    cv2.imwrite(str(registered_dir / "registered.png"), registered)

    metrics = {
        "source_keypoints": len(source_keypoints),
        "reference_keypoints": len(reference_keypoints),
        "good_matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse_pixels": rmse,
        "homography": homography.tolist(),
    }

    with open(metrics_dir / "results.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\n" + "-" * 60)
    print("Baseline Registration Complete.")
    print(f"  Registered image: {registered_dir / 'registered.png'}")
    print(f"  Matches visual:   {matches_dir / 'matches.png'}")
    print(f"  Metrics JSON:     {metrics_dir / 'results.json'}")
    print("-" * 60)


def run_adaptive_command(args):
    """Execute end-to-end adaptive registration."""
    print("=" * 65)
    print("Adaptive Lunar Registration Pipeline")
    print("=" * 65)
    print(f"Source:    {args.source}")
    print(f"Reference: {args.reference}")
    if args.gsd_source and args.gsd_reference:
        print(f"GSD:       Source={args.gsd_source} m/px | Reference={args.gsd_reference} m/px")

    out_dir = Path(args.output_dir)

    result = adaptive_register_images(
        source=args.source,
        reference=args.reference,
        gsd_source=args.gsd_source,
        gsd_reference=args.gsd_reference,
        output_dir=out_dir,
        force_model=args.force_model,
        enable_piecewise=not args.no_piecewise,
        apply_spatial_filter=args.spatial_filter,
    )

    print("\n" + "-" * 65)
    print("1. Pair Diagnostics:")
    chars = result["pair_characteristics"]
    print(f"  - Source Texture:       {chars['source_texture']['level']} (variance={chars['source_texture']['spatial_variance']:.1f})")
    print(f"  - Reference Texture:    {chars['reference_texture']['level']} (variance={chars['reference_texture']['spatial_variance']:.1f})")
    print(f"  - Illumination Shift:   {chars['illumination']['category']} (Bhattacharyya dist={chars['illumination']['bhattacharyya_distance']:.3f})")
    print(f"  - Relief Visual Proxy:  {chars['terrain_relief_proxy']['combined_complexity']} (mean edge density={chars['terrain_relief_proxy'].get('mean_edge_density', 0.0):.3f})")

    strat = result["strategy"]
    print("\n2. Adaptive Strategy Selected:")
    print(f"  - Representation:       {strat['representation']}")
    print(f"  - SIFT Feature Budget:  {strat['sift_n_features']} (contrast thresh={strat['sift_contrast_threshold']})")
    print(f"  - Lowe Ratio Cutoff:    {strat['matcher_ratio_threshold']}")
    print(f"  - Strategy Rationale:   {strat['rationale']}")

    print("\n3. Geometric Model Comparison:")
    print(f"  {'Model':<14} {'DOF':<5} {'Inliers':<9} {'Inlier Ratio':<14} {'RMSE (px)':<10}")
    print("  " + "-" * 54)
    for model_name, m_info in result["model_comparison"].items():
        is_selected = " [SELECTED]" if model_name == result["selected_model"] else ""
        print(f"  {model_name:<14} {m_info['dof']:<5} {m_info['inliers']:<9} {m_info['inlier_ratio']:<14.3f} {m_info['rmse']:<10.3f}{is_selected}")

    print("\n4. Registration Outcome:")
    print(f"  - Status:               {result['status']}")
    print(f"  - Model Used:           {result['selected_model'].upper()}")
    print(f"  - Inlier Matches:       {result['inliers']} / {result['matches_count']} ({result['inlier_ratio']*100:.1f}%)")
    print(f"  - Reprojection RMSE:    {result['rmse']:.4f} pixels")
    print(f"  - Piecewise Warp Used:  {result['piecewise_used']}")

    conf = result["confidence"]
    print("\n5. Explainable Confidence Assessment:")
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
        run_register_command(args.source, args.reference, output_dir_str=args.output_dir)
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