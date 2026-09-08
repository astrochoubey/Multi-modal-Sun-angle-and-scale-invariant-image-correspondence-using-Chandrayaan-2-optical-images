"""
Image-Pair Characterization Module for Lunar Remote Sensing.
Analyzes pair properties prior to registration:
- Texture strength (rich / medium / weak)
- Illumination difference (histogram divergence, shadow area discrepancy)
- Scale difference (spectral power distribution)
- Geometric displacement proxy
- Terrain complexity / relief proxy (gradient density, edge density)

NOTE: Relief proxy is strictly an observable image-based textural feature,
NOT true topographic elevation or Digital Elevation Model (DEM) data.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


def compute_texture_strength(image: np.ndarray) -> Dict[str, Any]:
    """
    Compute image texture strength using spatial variance, Shannon entropy, and gradient density.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))

    # Shannon entropy of normalized histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / (hist.sum() + 1e-7)
    non_zeros = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(non_zeros * np.log2(non_zeros)))

    # Gradient density
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    gradient_density = float(np.mean(mag > 20.0))

    if variance > 1200 and entropy > 6.5 and gradient_density > 0.15:
        level = "texture_rich"
    elif variance > 300 and entropy > 5.0:
        level = "medium_texture"
    else:
        level = "weak_texture"

    return {
        "level": level,
        "spatial_variance": round(variance, 2),
        "shannon_entropy": round(entropy, 3),
        "gradient_density": round(gradient_density, 4),
    }


def compute_illumination_difference(img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
    """
    Quantify illumination disparity via Bhattacharyya distance, mean luminance difference,
    and shadow area fraction discrepancy.
    """
    g1 = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 1. Mean and contrast disparity
    m1, s1 = float(np.mean(g1)), float(np.std(g1))
    m2, s2 = float(np.mean(g2)), float(np.std(g2))
    mean_diff = abs(m1 - m2)

    # 2. Histogram Bhattacharyya distance
    h1 = cv2.calcHist([g1], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [64], [0, 256])
    cv2.normalize(h1, h1, 1.0, 0, cv2.NORM_L1)
    cv2.normalize(h2, h2, 1.0, 0, cv2.NORM_L1)
    bhattacharyya = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

    # 3. Shadow fraction (DN < 30)
    shadow_fraction_1 = float(np.mean(g1 < 30))
    shadow_fraction_2 = float(np.mean(g2 < 30))
    shadow_delta = abs(shadow_fraction_1 - shadow_fraction_2)

    if bhattacharyya > 0.50 or shadow_delta > 0.20 or mean_diff > 45.0:
        category = "extreme_illumination_shift"
    elif bhattacharyya > 0.25 or shadow_delta > 0.08 or mean_diff > 20.0:
        category = "moderate_illumination_shift"
    else:
        category = "benign_illumination"

    return {
        "category": category,
        "bhattacharyya_distance": round(bhattacharyya, 4),
        "mean_intensity_delta": round(mean_diff, 2),
        "shadow_fraction_src": round(shadow_fraction_1, 4),
        "shadow_fraction_ref": round(shadow_fraction_2, 4),
        "shadow_disparity": round(shadow_delta, 4),
    }


def compute_spectral_scale_proxy(img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
    """
    Estimate effective frequency scale ratio from radial Fourier power spectra.
    """
    g1 = cv2.resize(img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (512, 512)).astype(np.float32)
    g2 = cv2.resize(img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (512, 512)).astype(np.float32)

    f1 = np.abs(np.fft.fftshift(np.fft.fft2(g1)))
    f2 = np.abs(np.fft.fftshift(np.fft.fft2(g2)))

    # High-frequency power ratio (outer 25% of spectrum)
    center = 256
    y, x = np.ogrid[:512, :512]
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    outer_mask = r > 180

    hf1 = float(np.mean(f1[outer_mask]))
    hf2 = float(np.mean(f2[outer_mask]))
    spectral_ratio = round(hf1 / (hf2 + 1e-6), 3)

    return {
        "spectral_high_frequency_ratio": spectral_ratio,
        "is_significant_scale_jump": bool(spectral_ratio > 2.5 or spectral_ratio < 0.4),
    }


def compute_terrain_relief_proxy(image: np.ndarray) -> Dict[str, Any]:
    """
    Estimate terrain visual complexity / relief proxy from spatial gradient and edge distributions.
    Clearly designated as an image-based proxy, NOT true 3D elevation.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge density
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Laplacian variance (sharpness and crater rim definition)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # Spatial variance across 4 quadrants to detect uneven relief
    h, w = gray.shape
    q1 = np.var(gray[:h//2, :w//2])
    q2 = np.var(gray[:h//2, w//2:])
    q3 = np.var(gray[h//2:, :w//2])
    q4 = np.var(gray[h//2:, w//2:])
    quadrant_spread = float(np.std([q1, q2, q3, q4]) / (np.mean([q1, q2, q3, q4]) + 1e-5))

    if edge_density > 0.08 or lap_var > 300.0 or quadrant_spread > 0.4:
        complexity = "high_relief_proxy"
    elif edge_density > 0.03 or lap_var > 80.0:
        complexity = "moderate_relief_proxy"
    else:
        complexity = "flat_mare_proxy"

    return {
        "complexity_class": complexity,
        "edge_density": round(edge_density, 4),
        "laplacian_variance": round(lap_var, 2),
        "quadrant_variance_spread": round(quadrant_spread, 3),
        "disclaimer": "Observable 2D image proxy; does not represent ground-truth elevation."
    }


def characterize_image_pair(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    metadata: Dict[str, Any] = None,
    gsd1: float = None,
    gsd2: float = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract comprehensive multi-factor characterization of an image pair.
    """
    meta = dict(metadata or {})
    if gsd1 is not None and gsd2 is not None and gsd2 > 0:
        meta["scale_ratio"] = float(gsd1 / gsd2)
    elif "gsd_source" in kwargs and "gsd_reference" in kwargs:
        s = kwargs["gsd_source"]
        r = kwargs["gsd_reference"]
        if s and r and r > 0:
            meta["scale_ratio"] = float(s / r)

    src_texture = compute_texture_strength(source_image)
    ref_texture = compute_texture_strength(reference_image)
    illumination = compute_illumination_difference(source_image, reference_image)
    scale_proxy = compute_spectral_scale_proxy(source_image, reference_image)
    src_relief = compute_terrain_relief_proxy(source_image)
    ref_relief = compute_terrain_relief_proxy(reference_image)

    # Known metadata scale ratio if supplied
    meta_scale = meta.get("scale_ratio", None)

    # Mean edge density for easy scalar reporting
    mean_edge_density = float(0.5 * (src_relief["edge_density"] + ref_relief["edge_density"]))

    return {
        "source_texture": src_texture,
        "reference_texture": ref_texture,
        "illumination": illumination,
        "scale": {
            **scale_proxy,
            "metadata_scale_ratio": meta_scale,
        },
        "terrain_relief_proxy": {
            "source": src_relief,
            "reference": ref_relief,
            "combined_complexity": max(src_relief["complexity_class"], ref_relief["complexity_class"]),
            "mean_edge_density": round(mean_edge_density, 4),
        }
    }
