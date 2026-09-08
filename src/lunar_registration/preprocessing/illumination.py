"""
Illumination and shadow correction filters for lunar imagery.
"""

import cv2
import numpy as np


def apply_wall_filter(image: np.ndarray, ksize: int = 31) -> np.ndarray:
    """
    Apply Wall filter (high-pass division) to suppress low-frequency illumination gradients.
    """
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-6) * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0).astype(np.float32) + 1.0
    quotient = (image.astype(np.float32) / blurred) * 128.0
    return np.clip(quotient, 0, 255).astype(np.uint8)


def remove_low_frequency_shadows(image: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Homomorphic / high-pass filtering to equalize illumination."""
    img_f = image.astype(np.float32) + 1.0
    log_img = np.log(img_f)
    low_freq = cv2.GaussianBlur(log_img, (0, 0), sigma)
    high_freq = log_img - low_freq
    exp_img = np.exp(high_freq)
    norm = cv2.normalize(exp_img, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)
