import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale if multi-channel, or return as-is.
    """
    if len(image.shape) == 2:
        return image
    if len(image.shape) == 3 and image.shape[2] == 1:
        return image.squeeze(2)
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

    if image.dtype != np.uint8:
        img_f = image.astype(np.float32)
        v_min, v_max = img_f.min(), img_f.max()
        if v_max > v_min:
            image = np.clip((img_f - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.zeros(image.shape, dtype=np.uint8)

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
    