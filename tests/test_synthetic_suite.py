"""
Unit tests for synthetic terrain generation and ground-truth evaluation.
"""

import numpy as np
import pytest

from lunar_registration.evaluation.synthetic_suite import (
    generate_synthetic_lunar_terrain,
    create_ground_truth_transform,
    evaluate_against_ground_truth,
    run_synthetic_benchmark,
)


def test_generate_synthetic_lunar_terrain():
    img = generate_synthetic_lunar_terrain(size=(128, 128), num_craters=15, seed=42)
    assert isinstance(img, np.ndarray)
    assert img.shape == (128, 128)
    assert img.dtype == np.uint8
    assert img.max() > img.min()


def test_create_ground_truth_transform():
    shape = (200, 200)
    # Translation
    H_trans = create_ground_truth_transform(shape, transform_type="translation", translation=(10.0, 5.0))
    assert H_trans.shape == (3, 3)
    assert np.isclose(H_trans[0, 2], 10.0)
    assert np.isclose(H_trans[1, 2], 5.0)

    # Similarity
    H_sim = create_ground_truth_transform(shape, transform_type="similarity", rotation_deg=10.0, scale=1.05)
    assert H_sim.shape == (3, 3)


def test_evaluate_against_ground_truth_identity():
    shape = (200, 200)
    H_eye = np.eye(3, dtype=np.float32)
    eval_res = evaluate_against_ground_truth(H_eye, H_eye, shape)
    assert eval_res["mean_corner_error_px"] == 0.0
    assert eval_res["mean_grid_error_px"] == 0.0


def test_run_synthetic_benchmark_small():
    results = run_synthetic_benchmark(output_dir=None, image_size=(128, 128))
    assert "pure_translation" in results
    assert "similarity_rot_scale" in results
    assert "representations" in results["pure_translation"]
