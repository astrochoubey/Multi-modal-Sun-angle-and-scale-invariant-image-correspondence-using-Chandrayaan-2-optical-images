"""
Piecewise / Grid-Based Local Registration for High-Relief Lunar Topography.
When 3D relief violates a single global planar homography, partitions
the scene into local tiles (H1...Hn), estimates local transformations,
and warps with feather blending.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class PiecewiseRegistrar:
    """
    Grid-based piecewise transformation estimator and warper.
    """

    def __init__(
        self,
        grid_rows: int = 3,
        grid_cols: int = 3,
        overlap_fraction: float = 0.25,
        min_cell_matches: int = 4,
        reproj_threshold: float = 3.0,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.overlap_fraction = overlap_fraction
        self.min_cell_matches = min_cell_matches
        self.reproj_threshold = reproj_threshold

    def fit(
        self,
        source_points: np.ndarray,
        reference_points: np.ndarray,
        reference_shape: tuple[int, int],
        global_H: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Estimate piecewise homographies per grid cell.
        """
        h_ref, w_ref = reference_shape[:2]
        cell_h = h_ref / float(self.grid_rows)
        cell_w = w_ref / float(self.grid_cols)

        tile_models = {}
        cell_stats = []

        pad_y = cell_h * self.overlap_fraction
        pad_x = cell_w * self.overlap_fraction

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y_min = max(0, r * cell_h - pad_y)
                y_max = min(h_ref, (r + 1) * cell_h + pad_y)
                x_min = max(0, c * cell_w - pad_x)
                x_max = min(w_ref, (c + 1) * cell_w + pad_x)

                # Find reference points in this cell region
                mask = (
                    (reference_points[:, 0] >= x_min) &
                    (reference_points[:, 0] <= x_max) &
                    (reference_points[:, 1] >= y_min) &
                    (reference_points[:, 1] <= y_max)
                )

                sub_src = source_points[mask]
                sub_ref = reference_points[mask]
                cell_pts = len(sub_src)

                if cell_pts >= self.min_cell_matches:
                    H_cell, inliers = cv2.findHomography(
                        sub_src, sub_ref, cv2.RANSAC, self.reproj_threshold
                    )
                    if H_cell is not None and inliers is not None and np.sum(inliers) >= 4:
                        used_H = H_cell
                        is_local = True
                    else:
                        used_H = global_H
                        is_local = False
                else:
                    used_H = global_H
                    is_local = False

                tile_models[(r, c)] = used_H
                cell_stats.append({
                    "grid_coord": (r, c),
                    "point_count": cell_pts,
                    "is_local_model": is_local,
                    "bounds": (int(x_min), int(y_min), int(x_max), int(y_max)),
                })

        return {
            "tile_models": tile_models,
            "cell_stats": cell_stats,
            "grid_dims": (self.grid_rows, self.grid_cols),
            "global_fallback_H": global_H,
        }

    def warp(
        self,
        source_image: np.ndarray,
        reference_shape: tuple[int, int],
        fitted_piecewise: Dict[str, Any],
    ) -> np.ndarray:
        """
        Warp source image using cell-specific transformations with smooth distance feathering.
        """
        h_ref, w_ref = reference_shape[:2]
        tile_models = fitted_piecewise["tile_models"]
        cell_h = h_ref / float(self.grid_rows)
        cell_w = w_ref / float(self.grid_cols)

        # Output accumulator and weight map for seamless feathering
        channels = source_image.shape[2] if len(source_image.shape) == 3 else 1
        if channels == 1 and len(source_image.shape) == 2:
            composite = np.zeros((h_ref, w_ref), dtype=np.float32)
        else:
            composite = np.zeros((h_ref, w_ref, channels), dtype=np.float32)

        weight_sum = np.zeros((h_ref, w_ref), dtype=np.float32)

        for (r, c), H_cell in tile_models.items():
            y_start = int(r * cell_h)
            y_end = int(min(h_ref, (r + 1) * cell_h))
            x_start = int(c * cell_w)
            x_end = int(min(w_ref, (c + 1) * cell_w))

            # Warp whole image under this local H
            warped_tile = cv2.warpPerspective(source_image, H_cell, (w_ref, h_ref))

            # Linear ramp weight across tile with smooth blend borders
            cell_weight = np.zeros((h_ref, w_ref), dtype=np.float32)
            cell_weight[y_start:y_end, x_start:x_end] = 1.0
            cell_weight_blurred = cv2.GaussianBlur(cell_weight, (21, 21), 0)

            if channels > 1:
                composite += warped_tile * cell_weight_blurred[..., np.newaxis]
            else:
                composite += warped_tile.astype(np.float32) * cell_weight_blurred

            weight_sum += cell_weight_blurred

        # Normalize by weight sum to avoid dark/bright seams
        weight_sum = np.maximum(weight_sum, 1e-5)
        if channels > 1:
            composite = composite / weight_sum[..., np.newaxis]
        else:
            composite = composite / weight_sum

        return np.clip(composite, 0, 255).astype(np.uint8)
