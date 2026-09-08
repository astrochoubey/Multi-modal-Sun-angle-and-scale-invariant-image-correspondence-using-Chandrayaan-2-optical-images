#!/usr/bin/env python3
"""
Lunar Image Registration - Classical SIFT Baseline Execution Utility

Role: Member E (Classical Baseline) & Member B (Evaluation)
Usage:
    python scripts/run_baseline.py --pair DEV-01 --config configs/sift.yaml
    python scripts/run_baseline.py --all --config configs/sift.yaml
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lunar_registration.config import load_config
from lunar_registration.io.loaders import load_image
from lunar_registration.io.writers import save_image, save_metrics
from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import calculate_rmse, calculate_inlier_ratio
from lunar_registration.evaluation.visualization import draw_matches


def run_pair(pair_row: dict, config: dict, output_dir: Path) -> dict:
    pair_id = pair_row["pair_id"]
    source_path = Path(pair_row["source_local_path"])
    ref_path = Path(pair_row["ref_local_path"])

    print(f"\n==========================================")
    print(f"Executing Baseline for Pair: {pair_id}")
    print(f"Source:    {source_path}")
    print(f"Reference: {ref_path}")
    print(f"==========================================")

    t0 = time.perf_counter()

    # 1. Load Images
    source = load_image(source_path)
    reference = load_image(ref_path)
    t_load = time.perf_counter()

    # 2. Preprocess
    source_proc = preprocess(source)
    ref_proc = preprocess(reference)
    t_prep = time.perf_counter()

    # 3. Detect Features
    n_features = config.get("features", {}).get("sift", {}).get("n_features", 5000)
    src_kp, src_desc = detect_and_compute(source_proc, n_features=n_features)
    ref_kp, ref_desc = detect_and_compute(ref_proc, n_features=n_features)
    t_feat = time.perf_counter()

    print(f"  Source Keypoints:    {len(src_kp)}")
    print(f"  Reference Keypoints: {len(ref_kp)}")

    if src_desc is None or ref_desc is None or len(src_kp) < 4 or len(ref_kp) < 4:
        print("  [ERROR] Insufficient keypoints extracted.")
        return {"pair_id": pair_id, "success": False, "reason": "insufficient_features"}

    # 4. Match Descriptors
    ratio_threshold = config.get("matching", {}).get("ratio_threshold", 0.75)
    matches = match_descriptors(src_desc, ref_desc, ratio_threshold=ratio_threshold)
    t_match = time.perf_counter()

    print(f"  Good Matches:        {len(matches)}")

    if len(matches) < 4:
        print("  [ERROR] Insufficient matches for homography estimation.")
        return {"pair_id": pair_id, "success": False, "reason": "insufficient_matches"}

    # 5. Estimate Geometry
    reproj_thresh = config.get("geometry", {}).get("reprojection_threshold_px", 3.0)
    H, inlier_mask = estimate_homography(src_kp, ref_kp, matches, reprojection_threshold=reproj_thresh)
    inliers_count = int(inlier_mask.sum())
    inlier_ratio = calculate_inlier_ratio(inlier_mask)
    t_geom = time.perf_counter()

    print(f"  Inliers:             {inliers_count}")
    print(f"  Inlier Ratio:        {inlier_ratio:.3f}")

    # 6. Warp & Metrics
    registered = warp_image(source, H, reference.shape)
    inlier_matches = [m for m, is_inl in zip(matches, inlier_mask) if is_inl]
    rmse = calculate_rmse(src_kp, ref_kp, inlier_matches, H)
    t_warp = time.perf_counter()

    print(f"  Reprojection RMSE:   {rmse:.4f} px")

    # 7. Spatial Coverage (8x8 grid)
    H_ref, W_ref = reference.shape[:2]
    grid_w, grid_h = W_ref / 8.0, H_ref / 8.0
    occupied_cells = set()
    for m in inlier_matches:
        pt = ref_kp[m.trainIdx].pt
        cell_x = min(int(pt[0] / grid_w), 7)
        cell_y = min(int(pt[1] / grid_h), 7)
        occupied_cells.add((cell_x, cell_y))
    spatial_coverage = len(occupied_cells) / 64.0
    print(f"  Spatial Coverage:    {spatial_coverage:.3f} ({len(occupied_cells)}/64 cells)")

    # 8. Save Artifacts
    matches_dir = output_dir / "matches"
    registered_dir = output_dir / "registered"
    metrics_dir = output_dir / "metrics"

    match_vis = draw_matches(source, src_kp, reference, ref_kp, matches, inlier_mask)
    save_image(match_vis, matches_dir / f"{pair_id.lower()}_matches.png")
    save_image(registered, registered_dir / f"{pair_id.lower()}_registered.png")

    result = {
        "pair_id": pair_id,
        "success": bool(inliers_count >= 15 and rmse <= 3.0 and spatial_coverage >= 0.25),
        "source_keypoints": len(src_kp),
        "reference_keypoints": len(ref_kp),
        "good_matches": len(matches),
        "inliers": inliers_count,
        "inlier_ratio": float(inlier_ratio),
        "reprojection_rmse_px": float(rmse),
        "spatial_coverage_score": float(spatial_coverage),
        "timing_ms": {
            "load": (t_load - t0) * 1000,
            "preprocess": (t_prep - t_load) * 1000,
            "features": (t_feat - t_prep) * 1000,
            "match": (t_match - t_feat) * 1000,
            "geometry": (t_geom - t_match) * 1000,
            "warp": (t_warp - t_geom) * 1000,
            "total": (t_warp - t0) * 1000,
        },
        "homography": H.tolist()
    }

    save_metrics(result, metrics_dir / f"{pair_id.lower()}_results.json")
    print(f"  Artifacts saved to {output_dir}/")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run classical registration baseline on pilot manifest")
    parser.add_argument("--manifest", default="data/manifests/pilot_manifest.csv", help="Path to pilot manifest")
    parser.add_argument("--config", default="configs/sift.yaml", help="Path to pipeline YAML config")
    parser.add_argument("--pair", default=None, help="Target specific pair ID (e.g. DEV-01)")
    parser.add_argument("--all", action="store_true", help="Execute over all pairs in manifest")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")

    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    target_rows = rows
    if args.pair:
        target_rows = [r for r in rows if r["pair_id"].upper() == args.pair.upper()]

    if not target_rows:
        print(f"[ERROR] No matching pairs found for '{args.pair}'")
        sys.exit(1)

    summary = []
    for r in target_rows:
        res = run_pair(r, config, output_dir)
        summary.append(res)

    print("\n==========================================")
    print("BASELINE EXECUTION SUMMARY")
    print("==========================================")
    for s in summary:
        status = "PASS" if s.get("success") else "FAIL"
        print(f"  Pair {s['pair_id']}: [{status}] | Inliers: {s.get('inliers', 0)} "
              f"| Ratio: {s.get('inlier_ratio', 0.0):.2f} | RMSE: {s.get('reprojection_rmse_px', 0.0):.3f} px "
              f"| Coverage: {s.get('spatial_coverage_score', 0.0):.2f}")


if __name__ == "__main__":
    main()
