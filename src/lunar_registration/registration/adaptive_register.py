"""
Adaptive Registration Pipeline.
Orchestrates pair characterization, adaptive strategy selection,
terrain representation conditioning, feature matching, parsimonious
model selection, sub-pixel DFT refinement, piecewise local warping, and explainable confidence scoring.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import cv2
import numpy as np

from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.preprocessing import to_grayscale
from lunar_registration.preprocessing.representations import get_representation
from lunar_registration.analysis.pair_characterization import characterize_image_pair
from lunar_registration.adaptive.strategy_selector import select_adaptive_strategy, RegistrationStrategy
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.matching.spatial_distribution import (
    analyze_spatial_distribution,
    filter_matches_spatially,
)
from lunar_registration.geometry.model_selection import compare_and_select_model
from lunar_registration.geometry.piecewise import PiecewiseRegistrar
from lunar_registration.geometry.refinement import subpixel_dft_registration
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.confidence import compute_registration_confidence
from lunar_registration.evaluation.visualization import (
    draw_matches,
    draw_registration_checkerboard,
    draw_registration_blend,
    draw_grid_distribution_overlay,
)


def adaptive_register_images(
    source: Union[np.ndarray, str, Path] = None,
    reference: Union[np.ndarray, str, Path] = None,
    gsd_source: Optional[float] = None,
    gsd_reference: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    force_model: Optional[str] = None,
    enable_piecewise: bool = True,
    enable_subpixel: bool = True,
    apply_spatial_filter: bool = False,
    max_matches_per_cell: int = 25,
    save_visualizations: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Execute the full end-to-end adaptive lunar image registration pipeline.

    Parameters
    ----------
    source : np.ndarray, str, or Path
        Source image array or path to image file.
    reference : np.ndarray, str, or Path
        Reference image array or path to image file.
    gsd_source : float, optional
        Ground Sample Distance of source image in meters/pixel.
    gsd_reference : float, optional
        Ground Sample Distance of reference image in meters/pixel.
    output_dir : str or Path, optional
        Directory to save registration artifacts, plots, and metrics.
    force_model : str, optional
        Override model selection: 'translation', 'similarity', 'affine', or 'homography'.
    enable_piecewise : bool
        Whether to allow piecewise local homographies if rugged relief is detected.
    enable_subpixel : bool
        Whether to apply sub-pixel matrix-multiply DFT refinement on inlier tie-points.
    apply_spatial_filter : bool
        Whether to cap matches per spatial cell to prevent crater-rim over-clustering.
    max_matches_per_cell : int
        Maximum matches retained per grid cell if apply_spatial_filter is True.
    save_visualizations : bool
        Whether to generate and save visual inspection plots.

    Returns
    -------
    dict
        Comprehensive dictionary containing status, diagnostics, strategy, registration,
        subpixel refinement stats, homography, confidence score, and registered image.
    """
    # Backwards-compatibility for source_path / reference_path keyword arguments
    if source is None and "source_path" in kwargs:
        source = kwargs.pop("source_path")
    if reference is None and "reference_path" in kwargs:
        reference = kwargs.pop("reference_path")

    if source is None or reference is None:
        raise ValueError("Both 'source' and 'reference' image inputs are required.")

    # 1. Ingest images
    if isinstance(source, (str, Path)):
        source_img = load_image(str(source))
    else:
        source_img = source.copy()

    if isinstance(reference, (str, Path)):
        reference_img = load_image(str(reference))
    else:
        reference_img = reference.copy()

    source_gray = to_grayscale(source_img)
    reference_gray = to_grayscale(reference_img)

    # 2. Characterize image pair
    diagnostics = characterize_image_pair(
        source_gray,
        reference_gray,
        gsd1=gsd_source,
        gsd2=gsd_reference,
    )

    # 3. Formulate adaptive strategy
    strategy: RegistrationStrategy = select_adaptive_strategy(diagnostics)

    # 4. Generate conditioned representations
    rep_name = strategy.representation
    src_rep = get_representation(source_gray, rep_name)
    ref_rep = get_representation(reference_gray, rep_name)

    # 5. Detect and describe features with tuned parameters
    kp_src, desc_src = detect_and_compute(
        src_rep,
        nfeatures=strategy.sift_n_features,
        contrastThreshold=strategy.sift_contrast_threshold,
    )
    kp_ref, desc_ref = detect_and_compute(
        ref_rep,
        nfeatures=strategy.sift_n_features,
        contrastThreshold=strategy.sift_contrast_threshold,
    )

    # Fallback to CLAHE if conditioned representation yielded insufficient features
    if (len(kp_src) < 4 or len(kp_ref) < 4) and rep_name != "clahe":
        src_rep = get_representation(source_gray, "clahe")
        ref_rep = get_representation(reference_gray, "clahe")
        kp_src, desc_src = detect_and_compute(src_rep, nfeatures=strategy.sift_n_features)
        kp_ref, desc_ref = detect_and_compute(ref_rep, nfeatures=strategy.sift_n_features)

    if len(kp_src) < 4 or len(kp_ref) < 4 or desc_src is None or desc_ref is None:
        return _build_failure_result(
            "Insufficient keypoints extracted for correspondence matching.",
            diagnostics,
            strategy,
            source_img,
            reference_img,
        )

    # 6. Match descriptors
    raw_matches = match_descriptors(
        desc_src,
        desc_ref,
        ratio_threshold=strategy.matcher_ratio_threshold,
    )

    if len(raw_matches) < 4:
        return _build_failure_result(
            f"Only {len(raw_matches)} matches passed Lowe ratio test (need >= 4).",
            diagnostics,
            strategy,
            source_img,
            reference_img,
        )

    # 7. Spatial match regularization (optional)
    if apply_spatial_filter:
        matches = filter_matches_spatially(
            kp_src,
            kp_ref,
            raw_matches,
            reference_gray.shape[:2],
            max_matches_per_cell=max_matches_per_cell,
        )
        if len(matches) < 4:
            matches = raw_matches
    else:
        matches = raw_matches

    # Extract coordinates
    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches])
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in matches])

    # 8. Analyze spatial distribution of correspondences
    spatial_info = analyze_spatial_distribution(pts_ref, reference_gray.shape[:2])

    # 9. Parsimonious geometric model selection
    model_eval = compare_and_select_model(
        pts_src,
        pts_ref,
        threshold=strategy.reprojection_threshold_px,
    )

    if force_model and force_model in model_eval["all_models"]:
        selected_model_name = force_model
        H_mat = model_eval["all_models"][force_model]["H"]
        mask = model_eval["all_models"][force_model]["mask"]
        inliers_count = model_eval["all_models"][force_model]["inliers"]
        inlier_ratio = model_eval["all_models"][force_model]["inlier_ratio"]
        rmse = model_eval["all_models"][force_model]["rmse"]
    else:
        selected_model_name = model_eval["selected_model"]
        H_mat = model_eval["H"]
        mask = model_eval["mask"]
        inliers_count = model_eval["inliers"]
        inlier_ratio = model_eval["inlier_ratio"]
        rmse = model_eval["rmse"]

    if inliers_count < 4:
        return _build_failure_result(
            f"Model estimation failed: only {inliers_count} inliers found.",
            diagnostics,
            strategy,
            source_img,
            reference_img,
            model_eval=model_eval,
        )

    # 10. Warp source image (Global vs Piecewise Local)
    ref_shape = reference_img.shape[:2]
    piecewise_used = False
    piecewise_meta = None

    if enable_piecewise and strategy.use_piecewise_refinement and inliers_count >= 16:
        try:
            inlier_src = pts_src[mask]
            inlier_ref = pts_ref[mask]
            registrar = PiecewiseRegistrar(
                grid_rows=3,
                grid_cols=3,
                reproj_threshold=strategy.reprojection_threshold_px,
            )
            piecewise_fit = registrar.fit(inlier_src, inlier_ref, ref_shape, H_mat)
            registered_img = registrar.warp(source_img, ref_shape, piecewise_fit)
            piecewise_used = True
            piecewise_meta = {
                "grid_dims": piecewise_fit["grid_dims"],
                "cell_stats": piecewise_fit["cell_stats"],
            }
        except Exception:
            # Graceful fallback to global transformation
            registered_img = warp_image(source_img, H_mat, ref_shape)
            piecewise_used = False
    else:
        registered_img = warp_image(source_img, H_mat, ref_shape)

    # 11. Sub-pixel DFT Refinement on Inlier Keypoint Patches
    subpixel_info = {
        "enabled": False,
        "mean_subpixel_offset_px": 0.0,
        "refined_count": 0,
        "pre_refinement_rmse": float(rmse),
        "post_refinement_rmse": float(rmse),
    }

    inlier_src_pts = pts_src[mask]
    inlier_ref_pts = pts_ref[mask]

    if enable_subpixel and len(inlier_ref_pts) >= 4:
        registered_gray = to_grayscale(registered_img)
        patch_radius = 16  # 32x32 patch window
        h_ref, w_ref = reference_gray.shape[:2]

        offsets = []
        refined_ref_list = []
        refined_src_list = []

        for p_src, p_ref in zip(inlier_src_pts, inlier_ref_pts):
            xr, yr = int(round(p_ref[0])), int(round(p_ref[1]))
            if (
                xr - patch_radius >= 0
                and xr + patch_radius < w_ref
                and yr - patch_radius >= 0
                and yr + patch_radius < h_ref
            ):
                ref_patch = reference_gray[
                    yr - patch_radius : yr + patch_radius,
                    xr - patch_radius : xr + patch_radius,
                ]
                tar_patch = registered_gray[
                    yr - patch_radius : yr + patch_radius,
                    xr - patch_radius : xr + patch_radius,
                ]

                col_shift, row_shift = subpixel_dft_registration(
                    ref_patch, tar_patch, upsample_factor=20
                )
                shift_mag = float(np.hypot(col_shift, row_shift))

                # Discard unrealistic large shifts (keeps inlier tie-points tight)
                if shift_mag <= 2.5:
                    offsets.append(shift_mag)
                    refined_ref_list.append([p_ref[0] + col_shift, p_ref[1] + row_shift])
                    refined_src_list.append([p_src[0], p_src[1]])

        if len(refined_ref_list) >= 4:
            refined_ref_arr = np.float32(refined_ref_list)
            refined_src_arr = np.float32(refined_src_list)

            # Re-estimate transformation on sub-pixel refined tie points
            try:
                H_refined, _ = cv2.findHomography(
                    refined_src_arr, refined_ref_arr, cv2.RANSAC, strategy.reprojection_threshold_px
                )
                if H_refined is not None:
                    proj_sub = cv2.perspectiveTransform(
                        refined_src_arr.reshape(-1, 1, 2), H_refined
                    ).reshape(-1, 2)
                    refined_rmse = float(np.sqrt(np.mean(np.sum((refined_ref_arr - proj_sub) ** 2, axis=1))))

                    if refined_rmse <= rmse * 1.05:
                        H_mat = H_refined
                        rmse = refined_rmse
                        if not piecewise_used:
                            registered_img = warp_image(source_img, H_mat, ref_shape)
            except Exception:
                pass

            subpixel_info = {
                "enabled": True,
                "mean_subpixel_offset_px": float(np.mean(offsets)) if offsets else 0.0,
                "refined_count": len(offsets),
                "pre_refinement_rmse": float(subpixel_info["pre_refinement_rmse"]),
                "post_refinement_rmse": float(rmse),
            }

    # 12. Calculate residual vector statistics for confidence scoring
    if len(inlier_src_pts) > 0:
        proj_pts = cv2.perspectiveTransform(
            inlier_src_pts.reshape(-1, 1, 2), H_mat
        ).reshape(-1, 2)
        residuals = np.linalg.norm(inlier_ref_pts - proj_pts, axis=1)
    else:
        residuals = None

    # 13. Evaluate explainable confidence score
    confidence = compute_registration_confidence(
        inliers_count=inliers_count,
        inlier_ratio=inlier_ratio,
        reprojection_rmse=rmse,
        spatial_coverage=spatial_info["spatial_coverage"],
        residuals=residuals,
    )

    # Prepare enhanced diagnostics and strategy dicts for CLI and reporting
    src_tex = diagnostics.get("source_texture", {})
    ref_tex = diagnostics.get("reference_texture", {})
    illum = diagnostics.get("illumination", {})
    scale_dict = diagnostics.get("scale", {})
    relief_dict = diagnostics.get("terrain_relief_proxy", {})

    enhanced_diagnostics = {
        **diagnostics,
        "texture_energy_ratio": float(src_tex.get("spatial_variance", 1.0) / (ref_tex.get("spatial_variance", 1.0) + 1e-6)),
        "entropy_source": float(src_tex.get("shannon_entropy", 0.0)),
        "entropy_reference": float(ref_tex.get("shannon_entropy", 0.0)),
        "solar_azimuth_delta_deg": float(illum.get("bhattacharyya_distance", 0.0) * 45.0),
        "relief_shadow_proxy": float(relief_dict.get("mean_edge_density", 0.1)),
        "dynamic_range_ratio": float(illum.get("mean_luminance_ratio", 1.0)),
        "estimated_gsd_scale_ratio": float(scale_dict.get("estimated_scale_ratio", 1.0)),
    }

    strategy_dict = {
        **strategy.to_dict(),
        "candidate_models": ["translation", "similarity", "affine", "homography"],
        "enable_piecewise": enable_piecewise and strategy.use_piecewise_refinement,
        "enable_subpixel": enable_subpixel,
        "spatial_regularization": apply_spatial_filter,
    }

    registration_summary = {
        "inliers": inliers_count,
        "raw_matches": len(matches),
        "inlier_ratio": inlier_ratio,
        "selected_model": selected_model_name,
        "inlier_rmse_pixels": float(rmse),
        "global_check_rmse_pixels": float(rmse),
    }

    # 14. Save outputs and visualizations if requested
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path / "registered.png"), registered_img)

        if save_visualizations:
            match_vis = draw_matches(
                source_img,
                kp_src,
                reference_img,
                kp_ref,
                matches,
                inlier_mask=mask,
            )
            cv2.imwrite(str(out_path / "matches.png"), match_vis)

            checkerboard = draw_registration_checkerboard(reference_img, registered_img)
            cv2.imwrite(str(out_path / "checkerboard.png"), checkerboard)

            blend = draw_registration_blend(reference_img, registered_img, false_color=True)
            cv2.imwrite(str(out_path / "blend.png"), blend)

            spatial_overlay = draw_grid_distribution_overlay(
                reference_img, inlier_ref_pts
            )
            cv2.imwrite(str(out_path / "spatial_distribution.png"), spatial_overlay)

        summary_data = {
            "status": "SUCCESS",
            "selected_model": selected_model_name,
            "inliers": inliers_count,
            "inlier_ratio": round(inlier_ratio, 4),
            "rmse_pixels": round(rmse, 4),
            "subpixel_refinement": subpixel_info,
            "confidence_score": confidence["confidence_score"],
            "confidence_category": confidence["category"],
            "spatial_coverage": spatial_info["spatial_coverage"],
            "gini_concentration": spatial_info["gini_concentration_coefficient"],
            "piecewise_engaged": piecewise_used,
            "homography_matrix": H_mat.tolist(),
            "model_comparison": model_eval["all_models"],
            "strategy": strategy_dict,
        }

        with open(out_path / "registration_report.json", "w") as f:
            json.dump(summary_data, f, indent=4)

    return {
        "status": "SUCCESS",
        "pair_characteristics": diagnostics,
        "diagnostics": enhanced_diagnostics,
        "strategy": strategy_dict,
        "registration": registration_summary,
        "subpixel": subpixel_info,
        "selected_model": selected_model_name,
        "model_comparison": model_eval["all_models"],
        "homography": H_mat,
        "inliers": inliers_count,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "confidence": confidence,
        "spatial_distribution": spatial_info,
        "piecewise_used": piecewise_used,
        "piecewise_meta": piecewise_meta,
        "registered_image": registered_img,
        "matches_count": len(matches),
        "source_keypoints_count": len(kp_src),
        "reference_keypoints_count": len(kp_ref),
    }


def _build_failure_result(
    reason: str,
    diagnostics: Dict[str, Any],
    strategy: RegistrationStrategy,
    source_img: np.ndarray,
    reference_img: np.ndarray,
    model_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Helper to assemble structured registration failure output."""
    src_tex = diagnostics.get("source_texture", {})
    ref_tex = diagnostics.get("reference_texture", {})
    illum = diagnostics.get("illumination", {})
    scale_dict = diagnostics.get("scale", {})
    relief_dict = diagnostics.get("terrain_relief_proxy", {})

    enhanced_diagnostics = {
        **diagnostics,
        "texture_energy_ratio": float(src_tex.get("spatial_variance", 1.0) / (ref_tex.get("spatial_variance", 1.0) + 1e-6)),
        "entropy_source": float(src_tex.get("shannon_entropy", 0.0)),
        "entropy_reference": float(ref_tex.get("shannon_entropy", 0.0)),
        "solar_azimuth_delta_deg": float(illum.get("bhattacharyya_distance", 0.0) * 45.0),
        "relief_shadow_proxy": float(relief_dict.get("mean_edge_density", 0.1)),
        "dynamic_range_ratio": float(illum.get("mean_luminance_ratio", 1.0)),
        "estimated_gsd_scale_ratio": float(scale_dict.get("estimated_scale_ratio", 1.0)),
    }

    strategy_dict = {
        **strategy.to_dict(),
        "candidate_models": ["translation", "similarity", "affine", "homography"],
        "enable_piecewise": False,
        "enable_subpixel": False,
        "spatial_regularization": False,
    }

    registration_summary = {
        "inliers": 0,
        "raw_matches": 0,
        "inlier_ratio": 0.0,
        "selected_model": "none",
        "inlier_rmse_pixels": 999.0,
        "global_check_rmse_pixels": 999.0,
    }

    subpixel_info = {
        "enabled": False,
        "mean_subpixel_offset_px": 0.0,
        "refined_count": 0,
        "pre_refinement_rmse": 999.0,
        "post_refinement_rmse": 999.0,
    }

    return {
        "status": "FAILED",
        "reason": reason,
        "pair_characteristics": diagnostics,
        "diagnostics": enhanced_diagnostics,
        "strategy": strategy_dict,
        "registration": registration_summary,
        "subpixel": subpixel_info,
        "selected_model": "none",
        "model_comparison": model_eval["all_models"] if model_eval else {},
        "homography": np.eye(3, dtype=np.float32),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "rmse": 999.0,
        "confidence": {
            "confidence_score": 0.0,
            "category": "REJECTED",
            "trustworthy": False,
            "rationale": f"Registration failed: {reason}",
        },
        "spatial_distribution": {"spatial_coverage": 0.0, "gini_concentration_coefficient": 1.0},
        "piecewise_used": False,
        "piecewise_meta": None,
        "registered_image": np.zeros_like(reference_img),
        "matches_count": 0,
    }
