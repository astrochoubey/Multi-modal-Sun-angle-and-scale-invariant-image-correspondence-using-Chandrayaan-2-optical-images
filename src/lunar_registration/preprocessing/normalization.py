"""
Image normalization routines for planetary high-dynamic-range data.
"""

import numpy as np


def percentile_normalize(
    image: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0
) -> np.ndarray:
    """
    Scale image dynamically based on lower and upper percentiles.
    Converts to uint8 in range [0, 255].
    """
    img_float = image.astype(np.float32)
    v_min, v_max = np.percentile(img_float, (p_low, p_high))
    if v_max <= v_min:
        v_min, v_max = img_float.min(), img_float.max()
    if v_max > v_min:
        norm = np.clip((img_float - v_min) / (v_max - v_min), 0.0, 1.0)
    else:
        norm = np.zeros_like(img_float)
    return (norm * 255.0).astype(np.uint8)


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize array strictly to [0.0, 1.0] float32."""
    img_float = image.astype(np.float32)
    v_min, v_max = img_float.min(), img_float.max()
    if v_max > v_min:
        return (img_float - v_min) / (v_max - v_min)
    return np.zeros_like(img_float)
