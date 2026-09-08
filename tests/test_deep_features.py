"""
Unit tests for Sub-Pixel DFT Refinement, SuperPoint, LoFTR, and CLI dispatcher.
"""

from pathlib import Path
import numpy as np
import cv2
import pytest

from lunar_registration.geometry.refinement import subpixel_dft_registration
from lunar_registration.features.superpoint import SuperPointExtractor
from lunar_registration.features.loftr import LoFTRMatcher
from lunar_registration.registration.adaptive_register import adaptive_register_images
from lunar_registration.evaluation.synthetic_suite import (
    generate_synthetic_lunar_terrain,
    create_ground_truth_transform,
)


def test_subpixel_dft_precision():
    """Verify sub-pixel matrix-multiply DFT recovers fractional shifts under 0.05 px."""
    np.random.seed(123)
    patch1 = np.random.randn(32, 32).astype(np.float32)

    # Shift patch by fractional translation using Fourier phase shift
    nr, nc = patch1.shape
    u = np.fft.fftfreq(nc).reshape(1, -1)
    v = np.fft.fftfreq(nr).reshape(-1, 1)
    dx_true, dy_true = 0.35, -0.25
    phase = np.exp(-2j * np.pi * (u * dx_true + v * dy_true))
    patch2 = np.real(np.fft.ifft2(np.fft.fft2(patch1) * phase)).astype(np.float32)

    dx_est, dy_est = subpixel_dft_registration(patch1, patch2, upsample_factor=20)
    err = np.hypot(dx_est - dx_true, dy_est - dy_true)

    assert err < 0.05, f"Sub-pixel error {err:.4f} px exceeds 0.05 px bound"


def test_adaptive_subpixel_refinement(tmp_path):
    """Verify adaptive_register_images executes subpixel refinement on inlier patches."""
    size = (200, 200)
    ref = generate_synthetic_lunar_terrain(size=size, sun_azimuth_deg=45.0, seed=42)
    H_gt = create_ground_truth_transform(size, transform_type="translation", translation=(5.0, -3.0))
    src = cv2.warpPerspective(ref, np.linalg.inv(H_gt), size)

    src_path = tmp_path / "src.png"
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(src_path), src)
    cv2.imwrite(str(ref_path), ref)

    res = adaptive_register_images(
        source=src_path,
        reference=ref_path,
        enable_subpixel=True,
    )

    assert res["status"] == "SUCCESS"
    assert "subpixel" in res
    assert res["subpixel"]["enabled"] is True
    assert res["subpixel"]["refined_count"] > 0
    assert res["subpixel"]["mean_subpixel_offset_px"] >= 0.0
    assert "registration" in res
    assert res["registration"]["selected_model"] != "none"


def test_superpoint_extractor_real_weights():
    """Verify SuperPointExtractor initializes and extracts 256D descriptors."""
    extractor = SuperPointExtractor(max_keypoints=300)
    img = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(img, (80, 80), 30, 200, -1)
    cv2.circle(img, (40, 40), 15, 255, -1)
    cv2.circle(img, (120, 110), 20, 180, -1)

    kps, descs = extractor.extract(img)

    assert len(kps) > 0
    assert descs is not None
    assert descs.shape[1] == 256
    assert descs.shape[0] == len(kps)


def test_loftr_matcher_real_weights():
    """Verify LoFTRMatcher performs dense matching between image pairs."""
    matcher = LoFTRMatcher(match_threshold=0.1, max_image_size=320)
    img0 = np.zeros((160, 160), dtype=np.uint8)
    cv2.circle(img0, (80, 80), 35, 220, -1)
    cv2.circle(img0, (50, 40), 18, 180, -1)

    # Translate slightly
    M = np.float32([[1, 0, 4], [0, 1, -2]])
    img1 = cv2.warpAffine(img0, M, (160, 160))

    pts0, pts1, conf = matcher.match(img0, img1)

    assert isinstance(pts0, np.ndarray)
    assert isinstance(pts1, np.ndarray)
    assert isinstance(conf, np.ndarray)
    assert pts0.shape[1] == 2
    assert pts1.shape[1] == 2
    assert len(pts0) == len(pts1) == len(conf)
