import cv2
import numpy as np


def detect_and_compute(
    image: np.ndarray,
    n_features: int = 5000,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    **kwargs,
):
    """
    Detect SIFT keypoints and compute descriptors.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image.
    n_features : int
        Maximum number of features.
    contrast_threshold : float
        The contrast threshold used to filter out weak features in semi-uniform regions.
    edge_threshold : float
        The threshold used to filter out edge-like features.

    Returns
    -------
    keypoints : list
        Detected SIFT keypoints.
    descriptors : np.ndarray
        SIFT descriptors.
    """
    nf = kwargs.get("nfeatures", n_features)
    ct = kwargs.get("contrastThreshold", contrast_threshold)
    et = kwargs.get("edgeThreshold", edge_threshold)

    sift = cv2.SIFT_create(
        nfeatures=nf,
        contrastThreshold=ct,
        edgeThreshold=et,
    )

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None,
    )

    return keypoints, descriptors