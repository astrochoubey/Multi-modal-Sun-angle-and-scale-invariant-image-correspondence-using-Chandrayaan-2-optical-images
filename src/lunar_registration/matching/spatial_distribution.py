"""
Spatial Match Distribution and Regularization Module.
Enforces uniform spatial distribution of tie points across the lunar surface:
- Grid occupancy and spatial coverage ratio
- Gini coefficient of spatial match concentration
- Spatially bucketed / adaptive non-maximal filtering to prevent crater-rim clustering.
"""

from typing import List, Tuple, Dict, Any
import cv2
import numpy as np


def analyze_spatial_distribution(
    keypoints: Any,
    image_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
) -> Dict[str, Any]:
    """
    Analyze the spatial distribution of keypoints or 2D coordinates across an N x M grid.
    """
    h, w = image_shape[:2]
    gx, gy = grid_size
    gw = w / float(gx)
    gh = h / float(gy)

    cell_counts = np.zeros((gy, gx), dtype=np.int32)

    for kp in keypoints:
        if hasattr(kp, "pt"):
            pt = kp.pt
        elif isinstance(kp, (list, tuple, np.ndarray)):
            pt = (float(kp[0]), float(kp[1]))
        else:
            continue
        cx = min(int(pt[0] / gw), gx - 1)
        cy = min(int(pt[1] / gh), gy - 1)
        cell_counts[cy, cx] += 1

    total_cells = gx * gy
    occupied_cells = int(np.sum(cell_counts > 0))
    coverage_ratio = float(occupied_cells / total_cells)

    # Gini coefficient of match inequality across cells
    counts_flat = cell_counts.ravel().astype(np.float64)
    if np.sum(counts_flat) > 0:
        counts_sorted = np.sort(counts_flat)
        n = len(counts_sorted)
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * counts_sorted)) / (n * np.sum(counts_sorted)) - (n + 1) / n)
    else:
        gini = 1.0

    return {
        "grid_size": grid_size,
        "total_cells": total_cells,
        "occupied_cells": occupied_cells,
        "spatial_coverage": round(coverage_ratio, 4),
        "gini_concentration_coefficient": round(max(0.0, gini), 4),
        "cell_counts_matrix": cell_counts.tolist(),
        "max_cell_matches": int(np.max(cell_counts)),
        "min_cell_matches": int(np.min(cell_counts)),
        "mean_cell_matches": round(float(np.mean(cell_counts)), 2),
    }


def filter_matches_spatially(
    keypoints_src: List[cv2.KeyPoint],
    keypoints_ref: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    reference_shape: tuple[int, int],
    grid_size: tuple[int, int] = (8, 8),
    max_matches_per_cell: int = 25,
) -> List[cv2.DMatch]:
    """
    Filter correspondences to cap the maximum number of matches per spatial cell,
    preventing over-concentration in high-contrast crater rims.
    """
    if len(matches) == 0:
        return []

    h, w = reference_shape[:2]
    gx, gy = grid_size
    gw = w / float(gx)
    gh = h / float(gy)

    # Sort matches by distance (highest quality first)
    sorted_matches = sorted(matches, key=lambda m: m.distance)

    bucketed_matches: Dict[Tuple[int, int], List[cv2.DMatch]] = {}

    for m in sorted_matches:
        pt = keypoints_ref[m.trainIdx].pt
        cx = min(int(pt[0] / gw), gx - 1)
        cy = min(int(pt[1] / gh), gy - 1)
        cell = (cy, cx)

        if cell not in bucketed_matches:
            bucketed_matches[cell] = []

        if len(bucketed_matches[cell]) < max_matches_per_cell:
            bucketed_matches[cell].append(m)

    spatially_filtered = []
    for cell_list in bucketed_matches.values():
        spatially_filtered.extend(cell_list)

    return spatially_filtered
