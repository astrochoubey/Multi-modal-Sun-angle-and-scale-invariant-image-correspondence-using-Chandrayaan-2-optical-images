import argparse
import json
from pathlib import Path
import time

import cv2
import yaml

from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
    calculate_spatial_coverage,
)
from lunar_registration.evaluation.visualization import draw_matches
from scripts.generate_sample_data import (
    generate_dataset,
    get_real_lunar_scenario,
    get_real_paths,
)


def load_config(config_path: str | Path | None) -> dict:
    """Load configuration from YAML file if provided."""
    defaults = {
        "feature_extraction": {"n_features": 5000},
        "preprocessing": {"clahe": {"clip_limit": 2.0}},
        "matching": {"ratio_threshold": 0.75},
        "geometry": {"ransac": {"reprojection_threshold": 5.0}},
        "outputs": {
            "base_dir": "outputs",
            "matches_dir": "outputs/matches",
            "registered_dir": "outputs/registered",
            "metrics_dir": "outputs/metrics",
        },
    }
    if not config_path:
        return defaults

    path = Path(config_path)
    if not path.exists():
        return defaults

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Merge with defaults
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_k not in cfg[k]:
                    cfg[k][sub_k] = sub_v
    return cfg


def register_images(source_path, reference_path, config_path=None):
    cfg = load_config(config_path)
    n_features = cfg.get("feature_extraction", {}).get("n_features", 5000)
    ratio_threshold = cfg.get("matching", {}).get("ratio_threshold", 0.75)
    reproj_thresh = cfg.get("geometry", {}).get("ransac", {}).get("reprojection_threshold", 5.0)

    print("Loading images...")
    source = load_image(source_path)
    reference = load_image(reference_path)

    print("Preprocessing...")
    source_processed = preprocess(source)
    reference_processed = preprocess(reference)

    print(f"Detecting SIFT features (max {n_features})...")
    source_keypoints, source_descriptors = detect_and_compute(
        source_processed, n_features=n_features
    )
    reference_keypoints, reference_descriptors = detect_and_compute(
        reference_processed, n_features=n_features
    )

    print(f"Source keypoints: {len(source_keypoints)}")
    print(f"Reference keypoints: {len(reference_keypoints)}")

    print("Matching descriptors...")
    matches = match_descriptors(
        source_descriptors,
        reference_descriptors,
        ratio_threshold=ratio_threshold,
    )

    print(f"Good matches: {len(matches)}")

    if len(matches) < 4:
        raise RuntimeError("Not enough matches for homography estimation.")

    print("Estimating homography with RANSAC...")
    homography, inlier_mask = estimate_homography(
        source_keypoints,
        reference_keypoints,
        matches,
        reprojection_threshold=reproj_thresh,
    )

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = calculate_inlier_ratio(inlier_mask)

    print(f"Inliers: {inlier_count}")
    print(f"Inlier ratio: {inlier_ratio:.3f}")

    print("Warping source image...")
    registered = warp_image(
        source,
        homography,
        reference.shape,
    )

    print("Calculating RMSE and spatial coverage...")
    inlier_matches = [
        match for match, is_inlier in zip(matches, inlier_mask) if is_inlier
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

    print(f"RMSE: {rmse:.4f} pixels")
    print(f"Spatial Coverage: {spatial_coverage:.2%}")

    output_dir = Path("outputs")
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

    # Save outputs (both README convention and PNG versions)
    cv2.imwrite(str(matches_dir / "matches.png"), match_visualization)
    cv2.imwrite(str(matches_dir / "correspondences.png"), match_visualization)

    cv2.imwrite(str(registered_dir / "registered.png"), registered)
    cv2.imwrite(str(registered_dir / "registered_image.tif"), registered)

    metrics = {
        "keypoints_source": len(source_keypoints),
        "keypoints_reference": len(reference_keypoints),
        "raw_matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "rmse": round(rmse, 4),
        "spatial_coverage": round(spatial_coverage, 4),
        "homography": homography.tolist(),
    }

    with open(metrics_dir / "results.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nRegistration complete.")
    print(f"Registered image: {registered_dir / 'registered_image.tif'}")
    print(f"Matches:          {matches_dir / 'correspondences.png'}")
    print(f"Metrics:          {metrics_dir / 'results.json'}")

    return metrics


def run_evaluation(config_path=None):
    """Run quantitative evaluation on the default dataset."""
    src_path = Path("data/source/source.png")
    ref_path = Path("data/reference/reference.png")
    if not src_path.exists() or not ref_path.exists():
        generate_dataset()
    print("=" * 60)
    print("           LUNAR REGISTRATION EVALUATION PIPELINE")
    print("=" * 60)
    metrics = register_images(src_path, ref_path, config_path=config_path)
    print("=" * 60)
    print("                     EVALUATION REPORT")
    print("=" * 60)
    for k, v in metrics.items():
        if k != "homography":
            print(f"  {k:22}: {v}")
    print("=" * 60)


def run_benchmark():
    """Run comprehensive benchmark across multiple real lunar illumination and scale scenarios."""
    print("=" * 70)
    print("       CHANDRAYAAN-2 LUNAR REGISTRATION BENCHMARK SUITE")
    print("       (Evaluated on Real Chandrayaan-2 Optical Imagery)")
    print("=" * 70)

    scenarios = [
        ("Chandrayaan-2 Real Observation Pair (Primary)", "primary"),
        ("Real Sun-Angle Illumination Invariance", "illumination"),
        ("Real Cross-Sensor Scale Invariance (OHRC vs TMC-2)", "scale"),
        ("Real High-Resolution Crater Basin Crop", "crater_crop"),
    ]

    benchmark_results = []
    temp_dir = Path("data/benchmark")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for name, scenario_id in scenarios:
        print(f"\n[BENCHMARK] Running: {name}...")
        t0 = time.time()
        src_img, ref_img = get_real_lunar_scenario(scenario_id)
        src_p = temp_dir / f"bench_{scenario_id}_src.png"
        ref_p = temp_dir / f"bench_{scenario_id}_ref.png"
        cv2.imwrite(str(src_p), src_img)
        cv2.imwrite(str(ref_p), ref_img)

        m = register_images(src_p, ref_p)
        elapsed = round(time.time() - t0, 2)
        m["scenario"] = name
        m["elapsed_sec"] = elapsed
        benchmark_results.append(m)

    print("\n" + "=" * 70)
    print("                    BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Scenario':<42} | {'Inliers':<7} | {'Ratio':<7} | {'RMSE (px)':<9} | {'Coverage':<8}"
    print(header)
    print("-" * len(header))
    for res in benchmark_results:
        print(f"{res['scenario'][:42]:<42} | {res['inliers']:<7} | {res['inlier_ratio']*100:>5.1f}% | {res['rmse']:>8.4f}  | {res['spatial_coverage']*100:>6.1f}%")
    print("=" * 70)

    # Save benchmark report
    out_path = Path("outputs/metrics/benchmark.json")
    with open(out_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"\nBenchmark results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Lunar Image Registration CLI (Chandrayaan-2 Baseline)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # 1. register
    register_parser = subparsers.add_parser("register", help="Register a pair of lunar images")
    register_parser.add_argument(
        "--source",
        required=True,
        help="Path to source image",
    )
    register_parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference image",
    )
    register_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )

    # 2. evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run quantitative evaluation on dataset")
    eval_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )

    # 3. benchmark
    subparsers.add_parser("benchmark", help="Run full multi-scenario benchmark")

    args = parser.parse_args()

    if args.command == "register":
        register_images(args.source, args.reference, config_path=args.config)
    elif args.command == "evaluate":
        run_evaluation(config_path=args.config)
    elif args.command == "benchmark":
        run_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()