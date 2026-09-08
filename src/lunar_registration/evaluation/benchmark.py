"""
Benchmark evaluation pipeline for batch processing manifest entries.
"""

from typing import List, Dict, Any
from pathlib import Path
import csv
import time

from lunar_registration.config import load_config
from lunar_registration.io.loaders import load_image
from lunar_registration.io.writers import save_image, save_metrics
from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import calculate_rmse, calculate_inlier_ratio


class RegistrationBenchmark:
    """Automated benchmark harness."""

    def __init__(self, config_path: str | Path, output_dir: str | Path = "outputs"):
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)

    def run_manifest(self, manifest_path: str | Path) -> List[Dict[str, Any]]:
        manifest_path = Path(manifest_path)
        results = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                res = self.evaluate_pair(row)
                results.append(res)
        return results

    def evaluate_pair(self, row: dict) -> Dict[str, Any]:
        pair_id = row["pair_id"]
        src = load_image(row["source_local_path"])
        ref = load_image(row["ref_local_path"])

        src_p = preprocess(src)
        ref_p = preprocess(ref)

        src_kp, src_desc = detect_and_compute(src_p)
        ref_kp, ref_desc = detect_and_compute(ref_p)

        if src_desc is None or ref_desc is None or len(src_kp) < 4 or len(ref_kp) < 4:
            return {"pair_id": pair_id, "success": False, "reason": "insufficient_features"}

        matches = match_descriptors(src_desc, ref_desc)
        if len(matches) < 4:
            return {"pair_id": pair_id, "success": False, "reason": "insufficient_matches"}

        H, inlier_mask = estimate_homography(src_kp, ref_kp, matches)
        inliers = int(inlier_mask.sum())
        inlier_ratio = calculate_inlier_ratio(inlier_mask)

        inlier_matches = [m for m, inl in zip(matches, inlier_mask) if inl]
        rmse = calculate_rmse(src_kp, ref_kp, inlier_matches, H)

        return {
            "pair_id": pair_id,
            "success": bool(inliers >= 15 and rmse <= 3.0),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "rmse": rmse,
        }
