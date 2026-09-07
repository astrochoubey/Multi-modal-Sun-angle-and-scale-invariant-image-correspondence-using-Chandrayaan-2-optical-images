#!/usr/bin/env python3
"""
Synthetic Baseline Audit Script (Task E-01)
Member E — Classical Registration and Invariance Engineer

This script validates the existing baseline execution chain:
load -> grayscale/CLAHE -> SIFT -> ratio matching -> homography -> warp -> metrics
using a controlled synthetic lunar surface with known ground-truth geometry
and simulated illumination variation.

NOTE: This is a synthetic code-sanity benchmark only. Per team playbook,
synthetic results must NOT be used to claim real lunar flight performance.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
)
from lunar_registration.evaluation.visualization import draw_matches


def generate_synthetic_lunar_surface(
    width: int = 800,
    height: int = 800,
    seed: int = 42,
) -> np.ndarray:
    """Generate a textured synthetic surface with simulated lunar craters."""
    np.random.seed(seed)
    noise = np.random.normal(120, 25, (height, width)).astype(np.float32)
    surface = cv2.GaussianBlur(noise, (51, 51), 15)
    
    crater_specs = [
        (200, 250, 45), (450, 200, 60), (600, 500, 75),
        (250, 550, 50), (400, 400, 35), (550, 320, 30),
        (150, 650, 25), (680, 150, 40), (320, 120, 20),
        (500, 680, 55), (120, 150, 35), (380, 620, 28),
    ]
    
    y_coords, x_coords = np.ogrid[:height, :width]
    for cx, cy, radius in crater_specs:
        dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        bowl = np.exp(-dist_sq / (2 * (radius * 0.7) ** 2)) * 60
        rim = np.exp(-((np.sqrt(dist_sq) - radius) ** 2) / (2 * (radius * 0.25) ** 2)) * 40
        angle_bias = (-(x_coords - cx) - (y_coords - cy)) / (radius + 1e-5)
        shading = np.clip(angle_bias, -1.0, 1.0) * 20 * np.exp(-dist_sq / (2 * radius ** 2))
        surface = surface - bowl + rim + shading

    micro_texture = np.random.normal(0, 8, (height, width)).astype(np.float32)
    surface = surface + micro_texture
    return np.clip(surface, 0, 255).astype(np.uint8)


def apply_ground_truth_transform(
    image: np.ndarray,
    rotation_deg: float = 8.0,
    scale: float = 1.06,
    tx: float = 20.0,
    ty: float = -15.0,
    illumination_gradient: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a known Euclidean/Affine transformation and optional illumination ramp."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    M_affine = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    M_affine[0, 2] += tx
    M_affine[1, 2] += ty
    
    H_gt = np.eye(3, dtype=np.float64)
    H_gt[:2, :] = M_affine
    
    transformed = cv2.warpAffine(
        image,
        M_affine,
        (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    
    if illumination_gradient:
        gradient = np.linspace(0.7, 1.25, w, dtype=np.float32)
        transformed_float = transformed.astype(np.float32) * gradient[np.newaxis, :]
        transformed = np.clip(transformed_float, 0, 255).astype(np.uint8)
        
    return transformed, H_gt


def main():
    out_dir = Path("outputs/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("================================================================")
    print("   TASK E-01: SYNTHETIC BASELINE AUDIT EXECUTION")
    print("   Member E: Classical Registration and Invariance Engineer")
    print("================================================================")
    
    t_start = time.perf_counter()
    
    print("\n[1/7] Synthesizing crater-textured lunar surface pair...")
    reference_raw = generate_synthetic_lunar_surface(800, 800, seed=42)
    source_raw, H_gt = apply_ground_truth_transform(
        reference_raw,
        rotation_deg=8.0,
        scale=1.06,
        tx=20.0,
        ty=-15.0,
        illumination_gradient=True,
    )
    
    cv2.imwrite(str(out_dir / "synthetic_reference.png"), reference_raw)
    cv2.imwrite(str(out_dir / "synthetic_source.png"), source_raw)
    print(f"  -> Saved reference and source images to {out_dir}")

    # 2. Preprocessing (converting synthetic 2D to 3D BGR as load_image would do)
    source_bgr = cv2.cvtColor(source_raw, cv2.COLOR_GRAY2BGR)
    reference_bgr = cv2.cvtColor(reference_raw, cv2.COLOR_GRAY2BGR)

    t_prep_0 = time.perf_counter()
    print("\n[2/7] Running preprocessing (Grayscale + CLAHE)...")
    source_proc = preprocess(source_bgr)
    reference_proc = preprocess(reference_bgr)
    t_prep = (time.perf_counter() - t_prep_0) * 1000
    print(f"  -> Preprocessing complete in {t_prep:.2f} ms")

    t_feat_0 = time.perf_counter()
    print("\n[3/7] Extracting SIFT keypoints & descriptors...")
    src_kps, src_descs = detect_and_compute(source_proc)
    ref_kps, ref_descs = detect_and_compute(reference_proc)
    t_feat = (time.perf_counter() - t_feat_0) * 1000
    print(f"  -> Source keypoints detected:    {len(src_kps)}")
    print(f"  -> Reference keypoints detected: {len(ref_kps)}")
    print(f"  -> Feature extraction complete in {t_feat:.2f} ms")

    t_match_0 = time.perf_counter()
    print("\n[4/7] Matching descriptors (BFMatcher + Lowe's ratio test)...")
    raw_matches = match_descriptors(src_descs, ref_descs)
    t_match = (time.perf_counter() - t_match_0) * 1000
    print(f"  -> Good matches retained (ratio <= 0.75): {len(raw_matches)}")
    print(f"  -> Descriptor matching complete in {t_match:.2f} ms")
    
    if len(raw_matches) < 4:
        raise RuntimeError(f"Audit failure: Not enough matches ({len(raw_matches)}) for homography!")

    t_geom_0 = time.perf_counter()
    print("\n[5/7] Estimating Homography with RANSAC...")
    H_est, inlier_mask = estimate_homography(
        src_kps,
        ref_kps,
        raw_matches,
        reprojection_threshold=3.0,
    )
    t_geom = (time.perf_counter() - t_geom_0) * 1000
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = calculate_inlier_ratio(inlier_mask)
    print(f"  -> Inliers: {inlier_count} / {len(raw_matches)}")
    print(f"  -> Inlier ratio: {inlier_ratio:.3f} ({inlier_ratio*100:.1f}%)")
    print(f"  -> Geometry estimation complete in {t_geom:.2f} ms")

    print("\n[6/7] Computing transformation error and warping source...")
    inlier_matches = [m for m, is_inlier in zip(raw_matches, inlier_mask) if is_inlier]
    reprojection_rmse = calculate_rmse(
        src_kps,
        ref_kps,
        inlier_matches,
        H_est,
    )
    
    H_prod = np.dot(H_est, H_gt)
    norm_identity_diff = float(np.linalg.norm(H_prod / H_prod[2, 2] - np.eye(3)))
    
    registered_source = warp_image(source_raw, H_est, reference_raw.shape)
    cv2.imwrite(str(out_dir / "synthetic_registered.png"), registered_source)
    
    ch_h, ch_w = reference_raw.shape
    tile_size = 80
    checkerboard = np.zeros_like(reference_raw)
    for r in range(0, ch_h, tile_size):
        for c in range(0, ch_w, tile_size):
            if ((r // tile_size) + (c // tile_size)) % 2 == 0:
                checkerboard[r:r+tile_size, c:c+tile_size] = reference_raw[r:r+tile_size, c:c+tile_size]
            else:
                checkerboard[r:r+tile_size, c:c+tile_size] = registered_source[r:r+tile_size, c:c+tile_size]
    cv2.imwrite(str(out_dir / "synthetic_checkerboard_overlay.png"), checkerboard)
    
    print("\n[7/7] Rendering match visualization...")
    vis_img = draw_matches(
        source_proc,
        src_kps,
        reference_proc,
        ref_kps,
        inlier_matches,
    )
    cv2.imwrite(str(out_dir / "synthetic_matches_audit.png"), vis_img)
    print(f"  -> Saved match visualization to {out_dir / 'synthetic_matches_audit.png'}")
    
    t_total = (time.perf_counter() - t_start) * 1000

    results = {
        "benchmark": "synthetic_sanity_audit_v1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": "800x800",
        "ground_truth_transform": {
            "rotation_deg": 8.0,
            "scale": 1.06,
            "translation": [20.0, -15.0],
            "illumination_ramp": True,
        },
        "metrics": {
            "source_keypoints": len(src_kps),
            "reference_keypoints": len(ref_kps),
            "raw_matches": len(raw_matches),
            "inlier_count": inlier_count,
            "inlier_ratio": round(inlier_ratio, 4),
            "reprojection_rmse_pixels": round(reprojection_rmse, 4),
            "matrix_identity_frobenius_norm": round(norm_identity_diff, 5),
        },
        "timings_ms": {
            "preprocessing": round(t_prep, 2),
            "feature_extraction": round(t_feat, 2),
            "descriptor_matching": round(t_match, 2),
            "homography_ransac": round(t_geom, 2),
            "total_execution": round(t_total, 2),
        },
        "artifacts_generated": [
            str(out_dir / "synthetic_reference.png"),
            str(out_dir / "synthetic_source.png"),
            str(out_dir / "synthetic_matches_audit.png"),
            str(out_dir / "synthetic_registered.png"),
            str(out_dir / "synthetic_checkerboard_overlay.png"),
        ],
    }

    metrics_file = out_dir / "synthetic_audit_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> Metrics JSON saved to {metrics_file}")
    
    print("\n================================================================")
    print("   AUDIT BENCHMARK SUMMARY (SYNTHETIC ONLY)")
    print(f"   - Inlier Ratio:      {inlier_ratio*100:.1f}% ({inlier_count}/{len(raw_matches)})")
    print(f"   - Inlier RMSE:       {reprojection_rmse:.3f} px")
    print(f"   - Pipeline Latency:  {t_total:.1f} ms")
    print(f"   - Code Sanity Gate:  PASSED (No runtime crashes)")
    print("================================================================")


if __name__ == "__main__":
    main()
