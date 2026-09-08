#!/usr/bin/env python3
"""
Lunar Image Registration - Benchmark Evaluation & Metrics Aggregator

Role: Member B (Evaluation Lead)
Usage:
    python scripts/evaluate.py --metrics-dir outputs/metrics
    python scripts/evaluate.py --output-summary docs/experiments.md
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def load_metric_files(metrics_dir: Path) -> List[Dict]:
    """Load all JSON metric files from directory."""
    results = []
    if not metrics_dir.exists():
        return results

    for fpath in sorted(metrics_dir.glob("*_results.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_file"] = fpath.name
                results.append(data)
        except Exception as e:
            print(f"[WARNING] Could not parse {fpath.name}: {e}")

    return results


def summarize_metrics(results: List[Dict]) -> None:
    """Print formatted evaluation summary adhering to EVALUATION_PROTOCOL.md."""
    print("\n" + "=" * 80)
    print("CHANDRAYAAN-2 REGISTRATION BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)

    if not results:
        print("[Notice] No metric files found in outputs/metrics/. Run a registration pipeline first.")
        return

    print(f"{'Pair ID':<10} {'Status':<8} {'Inliers':<10} {'Inlier Ratio':<14} {'RMSE (px)':<12} {'Coverage':<10} {'Total ms':<10}")
    print("-" * 80)

    success_count = 0
    total_inliers = 0
    ratios = []
    rmses = []
    coverages = []

    for r in results:
        pair_id = r.get("pair_id", "Unknown")
        status = "PASS" if r.get("success") else "FAIL"
        if r.get("success"):
            success_count += 1

        inliers = r.get("inliers", 0)
        total_inliers += inliers

        ratio = r.get("inlier_ratio", 0.0)
        ratios.append(ratio)

        rmse = r.get("reprojection_rmse_px", 0.0)
        if rmse > 0:
            rmses.append(rmse)

        cov = r.get("spatial_coverage_score", 0.0)
        coverages.append(cov)

        t_ms = r.get("timing_ms", {}).get("total", 0.0)

        print(f"{pair_id:<10} {status:<8} {inliers:<10} {ratio:<14.3f} {rmse:<12.4f} {cov:<10.3f} {t_ms:<10.1f}")

    print("-" * 80)
    n = len(results)
    sr = (success_count / n) * 100.0
    avg_ratio = sum(ratios) / n if n else 0.0
    avg_rmse = sum(rmses) / len(rmses) if rmses else 0.0
    avg_cov = sum(coverages) / n if n else 0.0

    print(f"Total Evaluated:          {n}")
    print(f"Registration Success:     {success_count}/{n} ({sr:.1f}%)")
    print(f"Mean Inlier Ratio:        {avg_ratio:.3f}")
    print(f"Mean Reprojection RMSE:   {avg_rmse:.4f} px (across converged pairs)")
    print(f"Mean Spatial Coverage:    {avg_cov:.3f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and report registration evaluation metrics")
    parser.add_argument("--metrics-dir", default="outputs/metrics", help="Directory containing JSON metrics")

    args = parser.parse_args()
    results = load_metric_files(Path(args.metrics_dir))
    summarize_metrics(results)


if __name__ == "__main__":
    main()
