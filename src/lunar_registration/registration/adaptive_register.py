"""
Adaptive Registration Pipeline.
Orchestrates pair characterization, adaptive strategy selection,
terrain representation conditioning, feature matching, parsimonious
model selection, piecewise local refinement, and explainable confidence scoring.
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
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.confidence import compute_registration_confidence
from lunar_registration.evaluation.visualization import (
    draw_matches,
    draw_registration_checkerboard,
    draw_registration_blend,
    draw_grid_distribution_overlay,
)


def adaptive_register_images(
    source: Union[np.ndarray, str, Path],
    reference: Union[np.ndarray, str, Path],
    gsd_source: Optional[float] = None,
    gsd_reference: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    force_model: Optional[str] = None,
    enable_piecewise: bool = True,
    apply_spatial_filter: bool = False,
    max_matches_per_cell: int = 25,
    save_visualizations: bool = True,
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
    apply_spatial_filter : bool
        Whether to cap matches per spatial cell to prevent crater-rim over-clustering.
    max_matches_per_cell : int
        Maximum matches retained per grid cell if apply_spatial_filter is True.
    save_visualizations : bool
        Whether to generate and save visual inspection plots.

    Returns
    -------
    dict
        Comprehensive dictionary containing:
        - status: 'SUCCESS' or 'FAILED'
        - pair_characteristics: Diagnostics from pair characterization
        - strategy: Recommended strategy parameters and rationale
        - selected_model: Chosen geometric model ('translation', 'similarity', 'affine', 'homography')
        - model_comparison: Comparative performance of all 4 geometric models
        - homography: 3x3 transformation matrix
        - inliers: Number of inlier correspondences
        - inlier_ratio: Inlier fraction
        - rmse: Reprojection RMSE in pixels
        - confidence: Composite confidence score and explainable breakdown
        - spatial_distribution: Grid occupancy, coverage ratio, and Gini coefficient
        - piecewise_used: Whether piecewise local warping was engaged
        - registered_image: Registered source image in reference coordinate frame
    """
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

    # 11. Calculate residual vector statistics for confidence scoring
    inlier_src_pts = pts_src[mask]
    inlier_ref_pts = pts_ref[mask]
    if len(inlier_src_pts) > 0:
        proj_pts = cv2.perspectiveTransform(
            inlier_src_pts.reshape(-1, 1, 2), H_mat
        ).reshape(-1, 2)
        residuals = np.linalg.norm(inlier_ref_pts - proj_pts, axis=1)
    else:
        residuals = None

    # 12. Evaluate explainable confidence score
    confidence = compute_registration_confidence(
        inliers_count=inliers_count,
        inlier_ratio=inlier_ratio,
        reprojection_rmse=rmse,
        spatial_coverage=spatial_info["spatial_coverage"],
        residuals=residuals,
    )

    # 13. Save outputs and visualizations if requested
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
            "confidence_score": confidence["confidence_score"],
            "confidence_category": confidence["category"],
            "spatial_coverage": spatial_info["spatial_coverage"],
            "gini_concentration": spatial_info["gini_concentration_coefficient"],
            "piecewise_engaged": piecewise_used,
            "homography_matrix": H_mat.tolist(),
            "model_comparison": model_eval["all_models"],
            "strategy": strategy.to_dict(),
        }

        with open(out_path / "registration_report.json", "w") as f:
            json.dump(summary_data, f, indent=4)

    return {
        "status": "SUCCESS",
        "pair_characteristics": diagnostics,
        "strategy": strategy.to_dict(),
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
    return {
        "status": "FAILED",
        "reason": reason,
        "pair_characteristics": diagnostics,
        "strategy": strategy.to_dict(),
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
