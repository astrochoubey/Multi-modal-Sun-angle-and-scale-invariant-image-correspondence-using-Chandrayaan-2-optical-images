"""
Synthetic Lunar Terrain Generator and Controlled Ground-Truth Benchmark Suite.
Generates simulated lunar surfaces with craters, shadows, and textures,
then applies known geometric and radiometric transformations to rigorously
measure registration error against absolute ground truth.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import json
import cv2
import numpy as np

from lunar_registration.preprocessing.representations import get_representation
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.model_selection import compare_and_select_model


def generate_synthetic_lunar_terrain(
    size: tuple[int, int] = (512, 512),
    num_craters: int = 45,
    sun_azimuth_deg: float = 45.0,
    sun_elevation_deg: float = 30.0,
    noise_sigma: float = 4.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a photorealistic synthetic lunar surface with craters, crater rims,
    shadowing based on sun angle, and micro-texture regolith noise.

    Parameters
    ----------
    size : tuple of int
        (height, width) of the generated image.
    num_craters : int
        Number of craters to generate with power-law size distribution.
    sun_azimuth_deg : float
        Sun azimuth angle in degrees (direction from which sunlight arrives).
    sun_elevation_deg : float
        Sun elevation angle above horizon in degrees.
    noise_sigma : float
        Standard deviation of Gaussian regolith noise.
    seed : int
        Random seed for exact reproducibility.

    Returns
    -------
    np.ndarray
        Synthetic lunar grayscale image (uint8).
    """
    rng = np.random.RandomState(seed)
    h, w = size

    # Base lunar elevation model (gentle undulating topography)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dem = np.zeros((h, w), dtype=np.float32)

    # Low-frequency broad elevation swells
    dem += 15.0 * np.sin(x_coords / 80.0) * np.cos(y_coords / 90.0)
    dem += 8.0 * np.sin(x_coords / 35.0 + y_coords / 40.0)

    # Add craters with power-law size distribution (many small, few large)
    radii = 10.0 / (rng.uniform(0.1, 1.0, size=num_craters) ** 0.8)
    radii = np.clip(radii, 6.0, min(h, w) // 4)

    for r in radii:
        cx = rng.uniform(r, w - r)
        cy = rng.uniform(r, h - r)

        # Distance from crater center
        dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

        # Crater bowl (spherical depression) + raised rim
        depth = r * 0.25
        rim_height = r * 0.08
        rim_width = r * 0.3

        # Depression inside bowl
        bowl_mask = dist <= r
        dem[bowl_mask] -= depth * (1.0 - (dist[bowl_mask] / r) ** 2)

        # Raised rim around edge
        rim_mask = (dist > r * 0.8) & (dist < r + rim_width)
        rim_profile = np.sin(np.pi * (dist[rim_mask] - r * 0.8) / (rim_width + 0.2 * r))
        dem[rim_mask] += rim_height * rim_profile

    # Calculate surface slopes (gradients)
    dy, dx = np.gradient(dem)

    # Sun direction vector
    az_rad = np.radians(sun_azimuth_deg)
    el_rad = np.radians(sun_elevation_deg)
    sun_x = np.cos(el_rad) * np.cos(az_rad)
    sun_y = np.cos(el_rad) * np.sin(az_rad)
    sun_z = np.sin(el_rad)

    # Surface normal vector (assuming height scale dz = 1.0)
    normal_x = -dx
    normal_y = -dy
    normal_z = np.ones_like(dem)
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    # Lambertian reflectance model: I = max(0, N . L)
    shading = normal_x * sun_x + normal_y * sun_y + normal_z * sun_z
    shading = np.clip(shading, 0.0, 1.0)

    # Base albedo with lunar regolith noise
    albedo = 130.0 + rng.normal(0.0, noise_sigma, size=(h, w))
    image = albedo * (0.15 + 0.85 * shading)

    return np.clip(image, 0, 255).astype(np.uint8)


def create_ground_truth_transform(
    image_shape: tuple[int, int],
    transform_type: str = "rigid",
    translation: tuple[float, float] = (15.0, -10.0),
    rotation_deg: float = 6.0,
    scale: float = 1.08,
    shear: tuple[float, float] = (0.05, 0.02),
    keystone: tuple[float, float] = (0.0002, 0.0001),
) -> np.ndarray:
    """
    Generate an exact 3x3 ground-truth transformation matrix centered at image center.
    """
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    T_to_center = np.array([
        [1.0, 0.0, -cx],
        [0.0, 1.0, -cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    T_from_center = np.array([
        [1.0, 0.0, cx + translation[0]],
        [0.0, 1.0, cy + translation[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if transform_type == "translation":
        M = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    elif transform_type == "rotation" or transform_type == "rigid":
        M = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)
    elif transform_type == "similarity":
        M = np.array([
            [scale * cos_t, -scale * sin_t, 0.0],
            [scale * sin_t,  scale * cos_t, 0.0],
            [0.0,            0.0,           1.0]
        ], dtype=np.float64)
    elif transform_type == "affine":
        shx, shy = shear
        M = np.array([
            [scale * cos_t + shx, -scale * sin_t, 0.0],
            [scale * sin_t,        scale * cos_t + shy, 0.0],
            [0.0,                  0.0,                 1.0]
        ], dtype=np.float64)
    elif transform_type == "homography" or transform_type == "perspective":
        shx, shy = shear
        p1, p2 = keystone
        M = np.array([
            [scale * cos_t + shx, -scale * sin_t, 0.0],
            [scale * sin_t,        scale * cos_t + shy, 0.0],
            [p1,                   p2,                  1.0]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    H_gt = T_from_center @ M @ T_to_center
    H_gt = H_gt / H_gt[2, 2]
    return H_gt.astype(np.float32)


def evaluate_against_ground_truth(
    estimated_H: np.ndarray,
    ground_truth_H: np.ndarray,
    image_shape: tuple[int, int],
) -> Dict[str, float]:
    """
    Compute corner transfer error and grid transfer error between estimated and ground-truth H.
    """
    h, w = image_shape[:2]

    # 4 image corners
    corners = np.array([
        [0.0, 0.0],
        [w, 0.0],
        [w, h],
        [0.0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)

    gt_corners = cv2.perspectiveTransform(corners, ground_truth_H).reshape(-1, 2)
    est_corners = cv2.perspectiveTransform(corners, estimated_H).reshape(-1, 2)

    corner_distances = np.linalg.norm(gt_corners - est_corners, axis=1)
    mean_corner_error = float(np.mean(corner_distances))
    max_corner_error = float(np.max(corner_distances))

    # Grid of 100 sample points
    gy, gx = np.mgrid[0:h:10j, 0:w:10j]
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()]).reshape(-1, 1, 2).astype(np.float32)

    gt_grid = cv2.perspectiveTransform(grid_pts, ground_truth_H).reshape(-1, 2)
    est_grid = cv2.perspectiveTransform(grid_pts, estimated_H).reshape(-1, 2)

    grid_errors = np.linalg.norm(gt_grid - est_grid, axis=1)
    mean_grid_error = float(np.mean(grid_errors))

    return {
        "mean_corner_error_px": round(mean_corner_error, 4),
        "max_corner_error_px": round(max_corner_error, 4),
        "mean_grid_error_px": round(mean_grid_error, 4),
    }


def run_synthetic_benchmark(
    output_dir: Optional[Union[str, Path]] = None,
    image_size: tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    """
    Run an end-to-end benchmark on controlled synthetic lunar pairs across:
    1. Pure Translation (2-DOF)
    2. Rotation + Scale (Similarity, 4-DOF)
    3. Affine Shear (6-DOF)
    4. Perspective Keystone Tilt (8-DOF)
    5. Illumination Angle Divergence (different Sun azimuth)
    """
    results = {}
    h, w = image_size

    # Base reference image
    ref_image = generate_synthetic_lunar_terrain(size=image_size, sun_azimuth_deg=45.0, seed=101)

    test_cases = [
        ("pure_translation", "translation", {"translation": (20.0, -15.0)}, 45.0),
        ("similarity_rot_scale", "similarity", {"translation": (10.0, 5.0), "rotation_deg": 8.0, "scale": 1.06}, 45.0),
        ("affine_shear", "affine", {"translation": (8.0, -4.0), "rotation_deg": 5.0, "scale": 1.04, "shear": (0.04, 0.02)}, 45.0),
        ("perspective_tilt", "perspective", {"translation": (5.0, 5.0), "rotation_deg": 3.0, "scale": 1.02, "keystone": (0.00015, 0.00008)}, 45.0),
        ("illumination_shift", "similarity", {"translation": (12.0, -8.0), "rotation_deg": 4.0, "scale": 1.03}, 135.0),
    ]

    for test_name, transform_type, kwargs, sun_az in test_cases:
        # Generate source image with desired sun angle
        if sun_az == 45.0:
            source_raw = ref_image
        else:
            # Different sun angle creates shadows cast in different direction
            source_raw = generate_synthetic_lunar_terrain(size=image_size, sun_azimuth_deg=sun_az, seed=101)

        # Apply ground-truth warp
        H_gt = create_ground_truth_transform(image_size, transform_type=transform_type, **kwargs)
        source_warped = cv2.warpPerspective(source_raw, np.linalg.inv(H_gt), (w, h))

        # Evaluate classical baseline (raw grayscale) vs adaptive representation (CLAHE / gradient)
        representations_to_test = ["raw", "clahe", "gradient"]
        case_rep_results = {}

        for rep in representations_to_test:
            src_prep = get_representation(source_warped, rep)
            ref_prep = get_representation(ref_image, rep)

            kp_s, desc_s = detect_and_compute(src_prep)
            kp_r, desc_r = detect_and_compute(ref_prep)

            if desc_s is None or desc_r is None or len(kp_s) < 4 or len(kp_r) < 4:
                case_rep_results[rep] = {
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "reprojection_rmse": 999.0,
                    "mean_corner_error_px": 999.0,
                    "status": "INSUFFICIENT_FEATURES",
                }
                continue

            matches = match_descriptors(desc_s, desc_r, ratio_threshold=0.75)
            if len(matches) < 4:
                case_rep_results[rep] = {
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "reprojection_rmse": 999.0,
                    "mean_corner_error_px": 999.0,
                    "status": "INSUFFICIENT_MATCHES",
                }
                continue

            pts_s = np.float32([kp_s[m.queryIdx].pt for m in matches])
            pts_r = np.float32([kp_r[m.trainIdx].pt for m in matches])

            model_res = compare_and_select_model(pts_s, pts_r, threshold=3.0)
            H_est = model_res["H"]
            inliers = model_res["inliers"]
            inlier_ratio = model_res["inlier_ratio"]
            rmse = model_res["rmse"]

            gt_eval = evaluate_against_ground_truth(H_est, H_gt, image_size)

            case_rep_results[rep] = {
                "inliers": inliers,
                "inlier_ratio": round(inlier_ratio, 3),
                "reprojection_rmse": round(rmse, 3),
                "selected_model": model_res["selected_model"],
                "mean_corner_error_px": gt_eval["mean_corner_error_px"],
                "mean_grid_error_px": gt_eval["mean_grid_error_px"],
                "status": "SUCCESS" if gt_eval["mean_corner_error_px"] < 5.0 else "SUBOPTIMAL",
            }

        results[test_name] = {
            "ground_truth_transform_type": transform_type,
            "sun_azimuth_difference_deg": abs(sun_az - 45.0),
            "representations": case_rep_results,
        }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        with open(out_p / "synthetic_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)

    return results
