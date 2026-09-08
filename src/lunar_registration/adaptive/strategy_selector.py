"""
Interpretable Adaptive Strategy Selection Engine.
Selects optimal representation, feature extraction tuning, and geometric model
based on measured physical characteristics of the lunar image pair.
"""

from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class RegistrationStrategy:
    representation: str
    feature_method: str
    sift_n_features: int
    sift_contrast_threshold: float
    matcher_ratio_threshold: float
    geometric_model: str            # 'translation', 'similarity', 'affine', 'homography', 'piecewise'
    reprojection_threshold_px: float
    use_piecewise_refinement: bool
    scale_downsample_factor: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_adaptive_strategy(pair_characteristics: Dict[str, Any]) -> RegistrationStrategy:
    """
    Select an adaptive registration strategy based on interpretable rules.
    """
    reasons = []

    # 1. Inspect Illumination
    illum = pair_characteristics.get("illumination", {})
    illum_category = illum.get("category", "benign_illumination")
    bhatt = illum.get("bhattacharyya_distance", 0.0)

    if illum_category == "extreme_illumination_shift" or bhatt > 0.45:
        representation = "gradient"
        reasons.append("Extreme illumination shift detected: selecting gradient magnitude to emphasize slope/rim structure over shadow-inverted intensities.")
        reproj_thresh = 4.0
    elif illum_category == "moderate_illumination_shift" or bhatt > 0.20:
        representation = "clahe"
        reasons.append("Moderate illumination difference: selecting CLAHE for local contrast normalization.")
        reproj_thresh = 3.0
    else:
        representation = "clahe"
        reasons.append("Benign illumination: standard CLAHE contrast conditioning selected.")
        reproj_thresh = 3.0

    # 2. Inspect Texture
    src_tex = pair_characteristics.get("source_texture", {}).get("level", "medium_texture")
    ref_tex = pair_characteristics.get("reference_texture", {}).get("level", "medium_texture")

    if src_tex == "weak_texture" or ref_tex == "weak_texture":
        representation = "local_contrast"
        n_features = 8000
        contrast_thresh = 0.02
        ratio_thresh = 0.80
        reasons.append("Weak texture detected (smooth lunar mare): selecting local contrast representation and lowering feature detection threshold.")
    else:
        n_features = 5000
        contrast_thresh = 0.03
        ratio_thresh = 0.75

    # 3. Inspect Scale Disparity
    scale_info = pair_characteristics.get("scale", {})
    meta_scale = scale_info.get("metadata_scale_ratio")
    spectral_scale = scale_info.get("spectral_high_frequency_ratio", 1.0)

    downsample_factor = 1.0
    if meta_scale is not None and meta_scale > 3.0:
        downsample_factor = float(meta_scale)
        reasons.append(f"Significant metadata scale divergence ({meta_scale:.1f}x): enabling pre-downsampling of high-res source.")
    elif scale_info.get("is_significant_scale_jump", False):
        downsample_factor = float(max(spectral_scale, 1.0 / (spectral_scale + 1e-5)))
        reasons.append(f"Spectral high-frequency disparity ({spectral_scale:.2f}): adjusting scale pyramid.")

    # 4. Inspect Terrain Relief Proxy
    relief = pair_characteristics.get("terrain_relief_proxy", {})
    relief_class = relief.get("combined_complexity", "moderate_relief_proxy")

    if relief_class == "high_relief_proxy":
        geometric_model = "homography"
        use_piecewise = True
        reasons.append("High terrain relief proxy (rugged crater rims/massifs): selecting Homography with piecewise grid refinement contingency.")
    else:
        geometric_model = "homography"
        use_piecewise = False
        reasons.append("Moderate/planar relief proxy: single global homography candidate selected.")

    return RegistrationStrategy(
        representation=representation,
        feature_method="sift",
        sift_n_features=n_features,
        sift_contrast_threshold=contrast_thresh,
        matcher_ratio_threshold=ratio_thresh,
        geometric_model=geometric_model,
        reprojection_threshold_px=reproj_thresh,
        use_piecewise_refinement=use_piecewise,
        scale_downsample_factor=downsample_factor,
        rationale=" | ".join(reasons)
    )
