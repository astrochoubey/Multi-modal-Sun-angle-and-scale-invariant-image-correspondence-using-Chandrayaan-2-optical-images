"""
Local contrast enhancement and detail amplification for lunar surface textures.
"""

import cv2
import numpy as np


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE to enhance local geomorphological detail."""
    if image.dtype != np.uint8:
        img_f = image.astype(np.float32)
        v_min, v_max = img_f.min(), img_f.max()
        if v_max > v_min:
            image = np.clip((img_f - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.zeros(image.shape, dtype=np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """Sharpen micro-crater edges using unsharp masking."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
