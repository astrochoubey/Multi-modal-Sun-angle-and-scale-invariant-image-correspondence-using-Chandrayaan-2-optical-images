"""
Modular representation extraction for lunar terrain imagery.
Provides representations that emphasize terrain structure over raw pixel appearance:
- raw_grayscale: standard normalized intensity
- clahe: local contrast equalization
- gradient_magnitude: first-order derivative highlighting slopes and rim transitions
- local_contrast: local standard deviation / variance normalized map
- laplacian: second-order derivative highlighting ridge lines and crater rims
- edge_map: binary/structural edge representation with soft dilation
- phase_congruency: frequency-domain structural phase proxy invariant to illumination direction
"""

from enum import Enum
from typing import Dict, Any, Callable
import cv2
import numpy as np


class RepresentationType(str, Enum):
    RAW = "raw"
    CLAHE = "clahe"
    GRADIENT = "gradient"
    LOCAL_CONTRAST = "local_contrast"
    LAPLACIAN = "laplacian"
    EDGES = "edges"
    PHASE_CONGRUENCY = "phase_congruency"


def ensure_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is 2D grayscale uint8 [0, 255]."""
    if len(image.shape) == 3:
        if image.shape[2] == 1:
            img = image.squeeze(2)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image.copy()

    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        v_min, v_max = img_f.min(), img_f.max()
        if v_max > v_min:
            img = np.clip((img_f - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.zeros(img.shape, dtype=np.uint8)

    return img


def extract_raw_grayscale(image: np.ndarray, **kwargs) -> np.ndarray:
    """Raw normalized grayscale representation."""
    return ensure_grayscale_uint8(image)


def extract_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    **kwargs
) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization representation."""
    gray = ensure_grayscale_uint8(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def extract_gradient_magnitude(
    image: np.ndarray,
    ksize: int = 3,
    use_scharr: bool = False,
    **kwargs
) -> np.ndarray:
    """
    First-order spatial gradient magnitude: ||grad(I)|| = sqrt(Ix^2 + Iy^2).
    Highlights crater rims, steep slope transitions, and ridge features.
    """
    gray = ensure_grayscale_uint8(image)
    if use_scharr:
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    else:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def extract_local_contrast(
    image: np.ndarray,
    kernel_size: int = 15,
    epsilon: float = 1e-3,
    **kwargs
) -> np.ndarray:
    """
    Local contrast representation: standard deviation normalized by local mean.
    Emphasizes subtle surface texture across both brightly lit slopes and deep shadows.
    """
    gray = ensure_grayscale_uint8(image).astype(np.float32)
    k = (kernel_size, kernel_size)
    mean = cv2.blur(gray, k)
    mean_sq = cv2.blur(gray ** 2, k)
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std = np.sqrt(variance)

    # Local Weber-contrast proxy: std / (mean + epsilon)
    local_contrast = std / (mean + epsilon)
    norm = cv2.normalize(local_contrast, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def extract_laplacian(
    image: np.ndarray,
    ksize: int = 3,
    blur_ksize: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Second-order Laplacian of Gaussian: |del^2(I)|.
    Highlights micro-crater boundaries and sharp circular rim geometries.
    """
    gray = ensure_grayscale_uint8(image)
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    lap_abs = np.abs(lap)
    norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def extract_edges(
    image: np.ndarray,
    low_thresh: int = 40,
    high_thresh: int = 120,
    dilate_ksize: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Structural edge representation using Canny detector with optional soft morphological dilation.
    Provides thin, salient crater and ridge contours.
    """
    gray = ensure_grayscale_uint8(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    if dilate_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def extract_phase_congruency(
    image: np.ndarray,
    n_scales: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Frequency-domain Phase Congruency approximation.
    Produces strictly positive structural responses invariant to illumination sign inversion.
    """
    gray = ensure_grayscale_uint8(image).astype(np.float32)
    h, w = gray.shape

    # Multi-scale bandpass filter bank
    response = np.zeros((h, w), dtype=np.float32)
    sigmas = [1.5, 3.0, 6.0][:n_scales]

    for s in sigmas:
        b1 = cv2.GaussianBlur(gray, (0, 0), s)
        b2 = cv2.GaussianBlur(gray, (0, 0), s * 2.0)
        band = np.abs(b1 - b2)
        response += band

    norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


REPRESENTATION_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    RepresentationType.RAW.value: extract_raw_grayscale,
    RepresentationType.CLAHE.value: extract_clahe,
    RepresentationType.GRADIENT.value: extract_gradient_magnitude,
    RepresentationType.LOCAL_CONTRAST.value: extract_local_contrast,
    RepresentationType.LAPLACIAN.value: extract_laplacian,
    RepresentationType.EDGES.value: extract_edges,
    RepresentationType.PHASE_CONGRUENCY.value: extract_phase_congruency,
}


def get_representation(
    image: np.ndarray,
    method: str = "clahe",
    **kwargs
) -> np.ndarray:
    """
    Extract a specified terrain-structure representation.

    Parameters
    ----------
    image : np.ndarray
        Input image (BGR or single-channel).
    method : str
        Representation name ('raw', 'clahe', 'gradient', 'local_contrast', 'laplacian', 'edges', 'phase_congruency').
    **kwargs
        Parameters passed to the specific representation function.

    Returns
    -------
    np.ndarray
        Processed representation as uint8 single-channel image.
    """
    key = method.lower().strip()
    if key not in REPRESENTATION_REGISTRY:
        raise ValueError(
            f"Unknown representation '{method}'. Available: {list(REPRESENTATION_REGISTRY.keys())}"
        )
    return REPRESENTATION_REGISTRY[key](image, **kwargs)
