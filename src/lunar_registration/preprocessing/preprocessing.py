import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to improve local contrast.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    clip_limit : float
        CLAHE contrast limit.
    tile_grid_size : tuple
        Size of local regions.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    return clahe.apply(image)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Complete Phase 1 preprocessing pipeline.
    """

    gray = to_grayscale(image)
    enhanced = apply_clahe(gray)

    return enhanced
    