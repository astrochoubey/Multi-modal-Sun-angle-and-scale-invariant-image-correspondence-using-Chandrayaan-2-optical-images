"""
Main entry point for Lunar Image Registration pipeline.
SIH Problem Statement ID: 26166
Title: Multi-modal, Sun angle and scale invariant image correspondence
       using Chandrayaan-2 optical images (OHRC, TMC and IIRS)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from lunar_registration.cli import register_images
from scripts.generate_sample_data import generate_dataset


def create_visual_comparison(
    reference_path: Path,
    registered_path: Path,
    output_path: Path,
) -> None:
    """
    Create a composite comparison figure showing:
    1. Reference Image
    2. Registered Source Image
    3. Alpha-blended Overlay
    4. Checkerboard Comparison
    """
    ref = cv2.imread(str(reference_path))
    reg = cv2.imread(str(registered_path))

    if ref is None or reg is None:
        return

    h, w = ref.shape[:2]
    reg = cv2.resize(reg, (w, h))

    # 1. 50-50 Blend
    blended = cv2.addWeighted(ref, 0.5, reg, 0.5, 0)

    # 2. Checkerboard overlay (8x8 grid)
    checkerboard = np.copy(ref)
    grid_size = 8
    cell_h, cell_w = h // grid_size, w // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 1:
                checkerboard[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = (
                    reg[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                )

    # Add labels
    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        labeled = img.copy()
        cv2.rectangle(labeled, (10, 10), (320, 48), (0, 0, 0), -1)
        cv2.putText(
            labeled,
            text,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return labeled

    top_row = np.hstack([add_label(ref, "Reference (Chandrayaan-2)"), add_label(reg, "Registered Source")])
    bot_row = np.hstack([add_label(blended, "50/50 Alpha Blend"), add_label(checkerboard, "Checkerboard Overlay")])
    grid = np.vstack([top_row, bot_row])

    cv2.imwrite(str(output_path), grid)


def print_banner():
    banner = """
=============================================================================
  LUNAR IMAGE REGISTRATION PIPELINE (Phase 1 Baseline)
  Chandrayaan-2 Optical Imagery Correspondence (OHRC / TMC-2 / IIRS)
=============================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="Run Lunar Image Registration Baseline Pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to source lunar image (default: data/source/source.png)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference lunar image (default: data/reference/reference.png)",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force re-generation of synthetic lunar crater test images",
    )

    args = parser.parse_args()
    print_banner()

    # Prioritize real lunar imagery if present
    if args.source:
        source_path = Path(args.source)
    elif Path("data/source/real_source.png").exists():
        source_path = Path("data/source/real_source.png")
    else:
        source_path = Path("data/source/source.png")

    if args.reference:
        reference_path = Path(args.reference)
    elif Path("data/reference/real_reference.png").exists():
        reference_path = Path("data/reference/real_reference.png")
    elif Path("data/source/real_refrence.png").exists():
        reference_path = Path("data/source/real_refrence.png")
    else:
        reference_path = Path("data/reference/reference.png")

    # If images do not exist or regeneration is requested, generate sample dataset
    if args.regenerate_data or not source_path.exists() or not reference_path.exists():
        print("[INFO] Lunar dataset not found or regeneration requested.")
        print("[INFO] Generating synthetic lunar surface data...")
        src, ref = generate_dataset(output_root="data")
        source_path = src
        reference_path = ref
        print("[INFO] Lunar dataset ready.\n")

    print(f"Source Image:    {source_path.resolve()}")
    print(f"Reference Image: {reference_path.resolve()}\n")

    # Run the registration pipeline
    register_images(source_path, reference_path)

    # Generate composite comparison figure
    comparison_path = Path("outputs/registered/comparison_grid.png")
    create_visual_comparison(reference_path, Path("outputs/registered/registered.png"), comparison_path)
    if comparison_path.exists():
        print(f"Comparison Grid: {comparison_path}")

    # Read and print summary metrics
    results_path = Path("outputs/metrics/results.json")
    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
        print("\n" + "=" * 60)
        print("                 EVALUATION METRICS")
        print("=" * 60)
        src_kp = data.get("keypoints_source", data.get("source_keypoints", "N/A"))
        ref_kp = data.get("keypoints_reference", data.get("reference_keypoints", "N/A"))
        raw_m = data.get("raw_matches", data.get("good_matches", "N/A"))
        inliers = data.get("inliers", "N/A")
        ratio = data.get("inlier_ratio", 0.0)
        rmse = data.get("rmse", data.get("rmse_pixels", 0.0))
        coverage = data.get("spatial_coverage", "N/A")

        print(f"  * Detected Source Keypoints:     {src_kp}")
        print(f"  * Detected Reference Keypoints:  {ref_kp}")
        print(f"  * Matches Found:                 {raw_m}")
        print(f"  * RANSAC Inlier Matches:         {inliers}")
        print(f"  * Inlier Ratio:                  {ratio * 100:.2f}%")
        print(f"  * Reprojection RMSE:             {rmse:.4f} pixels")
        if coverage != "N/A":
            print(f"  * Spatial Coverage:              {coverage * 100:.2f}%")
        print("=" * 60)

    print("\n[SUCCESS] Lunar image registration completed successfully.\n")


if __name__ == "__main__":
    main()
