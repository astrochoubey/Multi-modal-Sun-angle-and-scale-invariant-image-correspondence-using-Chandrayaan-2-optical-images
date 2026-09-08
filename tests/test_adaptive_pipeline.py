"""
Unit tests for the end-to-end adaptive registration pipeline.
"""

from pathlib import Path
import numpy as np
import pytest

from lunar_registration.evaluation.synthetic_suite import (
    generate_synthetic_lunar_terrain,
    create_ground_truth_transform,
)
from lunar_registration.registration.adaptive_register import adaptive_register_images


@pytest.fixture
def synthetic_lunar_pair(tmp_path):
    size = (200, 200)
    ref = generate_synthetic_lunar_terrain(size=size, sun_azimuth_deg=45.0, seed=42)
    # Apply small shift
    H_gt = create_ground_truth_transform(size, transform_type="translation", translation=(8.0, -5.0))
    import cv2
    src = cv2.warpPerspective(ref, np.linalg.inv(H_gt), size)

    src_path = tmp_path / "src.png"
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(src_path), src)
    cv2.imwrite(str(ref_path), ref)

    return str(src_path), str(ref_path)


def test_adaptive_register_images_end_to_end(synthetic_lunar_pair, tmp_path):
    src_path, ref_path = synthetic_lunar_pair
    out_dir = tmp_path / "adaptive_out"

    res = adaptive_register_images(
        source=src_path,
        reference=ref_path,
        output_dir=out_dir,
        enable_piecewise=True,
    )

    assert res["status"] == "SUCCESS"
    assert res["inliers"] >= 4
    assert res["rmse"] < 3.0
    assert "selected_model" in res
    assert "confidence" in res
    assert "spatial_distribution" in res
    assert (out_dir / "registered.png").exists()
    assert (out_dir / "registration_report.json").exists()
