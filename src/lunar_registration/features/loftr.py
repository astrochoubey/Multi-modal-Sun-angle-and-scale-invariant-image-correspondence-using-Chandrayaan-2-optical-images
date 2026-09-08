"""
LoFTR (Detector-Free Local Feature Matching with Transformers) adapter.
"""

from typing import Tuple, Dict, Any
import numpy as np


class LoFTRMatcher:
    """
    Detector-free dense matching adapter for textureless lunar mare regions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.match_threshold = self.config.get("match_threshold", 0.2)

    def match(
        self,
        source_image: np.ndarray,
        reference_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce dense correspondences (source_pts, ref_pts, confidences).
        """
        # Placeholder adapter ready for torch.hub LoFTR integration
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
