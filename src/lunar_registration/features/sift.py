import cv2
import numpy as np


def detect_and_compute(
    image: np.ndarray,
    n_features: int = 5000,
):
    """
    Detect SIFT keypoints and compute descriptors.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image.

    n_features : int
        Maximum number of features.

    Returns
    -------
    keypoints : list
        Detected SIFT keypoints.

    descriptors : np.ndarray
        SIFT descriptors.
    """

    sift = cv2.SIFT_create(nfeatures=n_features)

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None,
    )

    return keypoints, descriptors