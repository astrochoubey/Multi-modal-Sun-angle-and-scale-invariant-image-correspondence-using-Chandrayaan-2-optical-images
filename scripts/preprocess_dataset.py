#!/usr/bin/env python3
"""
Lunar Image Registration - Dataset Preprocessing & Standardization Utility

Role: Member D (Data Lead) & Member E (Classical Baseline)
Usage:
    python scripts/preprocess_dataset.py --manifest data/manifests/pilot_manifest.csv
    python scripts/preprocess_dataset.py --pair DEV-01
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


PROCESSED_DIR = Path("data/processed")


def robust_percentile_normalize(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Robust percentile normalization to handle planetary high-dynamic-range imagery.
    Scales the 1st to 99th percentile range to [0, 255] uint8.
    """
    img_float = img.astype(np.float32)
    v_min, v_max = np.percentile(img_float, (p_low, p_high))
    if v_max <= v_min:
        v_min, v_max = img_float.min(), img_float.max()
    if v_max > v_min:
        norm = np.clip((img_float - v_min) / (v_max - v_min), 0.0, 1.0)
    else:
        norm = np.zeros_like(img_float)
    return (norm * 255.0).astype(np.uint8)


def load_raw_planetary_raster(file_path: Path, expected_shape: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """
    Load an image from disk, supporting raw binary PDS rasters (.img/.IMG) and standard image formats.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # If standard raster
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # Binary raw raster fallback
    file_bytes = file_path.stat().st_size
    num_pixels = expected_shape[0] * expected_shape[1]

    if file_bytes >= num_pixels * 2:
        # 16-bit unsigned integer
        raw = np.fromfile(str(file_path), dtype=">u2")  # Big-endian common in PDS
        if len(raw) < num_pixels:
            raw = np.fromfile(str(file_path), dtype="<u2")
        img = raw[:num_pixels].reshape(expected_shape)
    else:
        # 8-bit unsigned integer
        raw = np.fromfile(str(file_path), dtype=np.uint8)
        img = raw[:num_pixels].reshape(expected_shape)

    return img


def preprocess_pair(pair_row: dict, output_dir: Path = PROCESSED_DIR) -> dict:
    """
    Process a single manifest pair: load raw data, normalize contrast, and save standardized rasters.
    """
    pair_id = pair_row["pair_id"]
    src_path = Path(pair_row["source_local_path"])
    ref_path = Path(pair_row["ref_local_path"])

    print(f"\n[Preprocessing] Processing {pair_id}...")
    print(f"  Source:    {src_path}")
    print(f"  Reference: {ref_path}")

    src_raw = load_raw_planetary_raster(src_path)
    ref_raw = load_raw_planetary_raster(ref_path)

    # Apply percentile normalization
    src_norm = robust_percentile_normalize(src_raw)
    ref_norm = robust_percentile_normalize(ref_raw)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    src_enhanced = clahe.apply(src_norm)
    ref_enhanced = clahe.apply(ref_norm)

    # Save to data/processed
    output_dir.mkdir(parents=True, exist_ok=True)
    out_src = output_dir / f"{pair_id.lower()}_source_clahe.png"
    out_ref = output_dir / f"{pair_id.lower()}_ref_clahe.png"

    cv2.imwrite(str(out_src), src_enhanced)
    cv2.imwrite(str(out_ref), ref_enhanced)

    print(f"  [Output] Saved: {out_src} (Shape: {src_enhanced.shape})")
    print(f"  [Output] Saved: {out_ref} (Shape: {ref_enhanced.shape})")

    return {
        "pair_id": pair_id,
        "processed_source": str(out_src),
        "processed_reference": str(out_ref),
        "source_shape": src_enhanced.shape,
        "ref_shape": ref_enhanced.shape,
    }


def preprocess_all(manifest_path: Path, target_pair: str = None) -> list:
    """Preprocess all or a specific pair from manifest."""
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    results = []
    with open(manifest_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if target_pair and row["pair_id"].upper() != target_pair.upper():
                continue
            res = preprocess_pair(row)
            results.append(res)

    print(f"\n[SUCCESS] Preprocessed {len(results)} image pairs into {PROCESSED_DIR}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Planetary Dataset Preprocessing Utility")
    parser.add_argument("--manifest", default="data/manifests/pilot_manifest.csv", help="Path to pilot manifest")
    parser.add_argument("--pair", default=None, help="Target pair ID (e.g. DEV-01)")

    args = parser.parse_args()
    preprocess_all(Path(args.manifest), args.pair)


if __name__ == "__main__":
    main()
