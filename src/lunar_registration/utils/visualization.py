"""
Visualization and diagnostics plotting utilities.
"""

from typing import Tuple
import cv2
import numpy as np


def create_checkerboard(
    img1: np.ndarray,
    img2: np.ndarray,
    tile_size: int = 64
) -> np.ndarray:
    """
    Create an interleaved checkerboard pattern between two co-registered images
    to visually inspect registration accuracy along crater walls.
    """
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h, w = img1.shape[:2]
    checkerboard = img1.copy()

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if ((y // tile_size) + (x // tile_size)) % 2 == 1:
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                checkerboard[y:y_end, x:x_end] = img2[y:y_end, x:x_end]

    return checkerboard


def create_alpha_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Create an alpha-blended overlay of two aligned images."""
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.addWeighted(img1, alpha, img2, 1.0 - alpha, 0)
