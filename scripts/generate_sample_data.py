"""
Data preparation and scenario module for Chandrayaan-2 Real Lunar Imagery.
All data pairs are loaded and derived directly from real lunar observations.
"""

from pathlib import Path
import cv2
import numpy as np


def get_real_paths(root: str | Path = "data") -> tuple[Path, Path]:
    """Resolve paths to real source and reference lunar images."""
    root = Path(root)

    # Source candidate paths
    src_candidates = [
        root / "source" / "real_source.png",
        root / "source" / "source.png",
    ]
    src_path = next((p for p in src_candidates if p.exists()), None)

    # Reference candidate paths
    ref_candidates = [
        root / "reference" / "real_reference.png",
        root / "reference" / "real_refrence.png",
        root / "source" / "real_refrence.png",
        root / "reference" / "reference.png",
    ]
    ref_path = next((p for p in ref_candidates if p.exists()), None)

    if src_path is None or ref_path is None:
        raise FileNotFoundError(
            f"Could not locate real lunar imagery in {root}. "
            "Please ensure real_source.png and real_reference.png exist."
        )

    return src_path, ref_path


def load_real_pair(root: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Load real Chandrayaan-2 source and reference lunar images."""
    src_path, ref_path = get_real_paths(root)
    src = cv2.imread(str(src_path))
    ref = cv2.imread(str(ref_path))

    if src is None or ref is None:
        raise FileNotFoundError(f"Failed to load real lunar images from {src_path} and {ref_path}")

    return src, ref


def get_real_lunar_scenario(scenario_id: str = "primary", root: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """
    Generate test scenarios derived directly from real Chandrayaan-2 lunar imagery:
    1. 'primary': Full unmodified real lunar observation pair.
    2. 'illumination': Real imagery under solar angle gradient and shadow variation.
    3. 'scale': Real imagery with multi-scale sensor simulation (OHRC vs TMC-2).
    4. 'crater_crop': High-relief impact crater basin cropped from the real lunar image.
    """
    src, ref = load_real_pair(root)
    h, w = src.shape[:2]

    if scenario_id == "primary":
        return src, ref

    elif scenario_id == "illumination":
        # Simulate cross-pass solar elevation shift across real lunar surface
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        az_rad = np.radians(45.0)
        grad = 0.82 + 0.36 * (np.cos(az_rad) * (x / w) + np.sin(az_rad) * (y / h))
        src_illum = np.clip(src.astype(np.float32) * grad[:, :, None], 0, 255).astype(np.uint8)
        return src_illum, ref

    elif scenario_id == "scale":
        # Simulate cross-sensor GSD difference (TMC-2 vs OHRC) and 4° camera orientation change
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle=4.0, scale=0.92)
        src_scale = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return src_scale, ref

    elif scenario_id in ["crater_crop", "complex_crater"]:
        # Focused high-resolution crater structure cropped from real imagery
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        return src[y1:y2, x1:x2], ref[y1:y2, x1:x2]

    return src, ref


def generate_dataset(output_root: str | Path = "data") -> tuple[Path, Path]:
    """Ensure data/source/source.png and data/reference/reference.png point to real data."""
    output_root = Path(output_root)
    src_path, ref_path = get_real_paths(output_root)

    dest_src = output_root / "source" / "source.png"
    dest_ref = output_root / "reference" / "reference.png"
    dest_raw = output_root / "raw" / "reference.png"

    dest_src.parent.mkdir(parents=True, exist_ok=True)
    dest_ref.parent.mkdir(parents=True, exist_ok=True)
    dest_raw.parent.mkdir(parents=True, exist_ok=True)

    if not dest_src.exists() or dest_src.resolve() != src_path.resolve():
        src_img = cv2.imread(str(src_path))
        if src_img is not None:
            cv2.imwrite(str(dest_src), src_img)

    if not dest_ref.exists() or dest_ref.resolve() != ref_path.resolve():
        ref_img = cv2.imread(str(ref_path))
        if ref_img is not None:
            cv2.imwrite(str(dest_ref), ref_img)
            cv2.imwrite(str(dest_raw), ref_img)

    return dest_src, dest_ref


if __name__ == "__main__":
    s, r = generate_dataset()
    print(f"Verified Real Lunar Dataset:\n  Source:    {s}\n  Reference: {r}")
