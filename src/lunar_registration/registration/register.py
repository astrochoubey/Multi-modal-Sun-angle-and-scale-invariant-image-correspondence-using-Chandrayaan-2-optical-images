import cv2
import numpy as np


def warp_image(
    source_image: np.ndarray,
    homography: np.ndarray,
    reference_shape: tuple[int, int],
) -> np.ndarray:
    """
    Warp the source image into the reference image coordinate system.

    Parameters
    ----------
    source_image : np.ndarray
        Source image.

    homography : np.ndarray
        3x3 homography matrix.

    reference_shape : tuple
        Reference image shape as (height, width).

    Returns
    -------
    np.ndarray
        Registered image.
    """

    height, width = reference_shape[:2]

    registered = cv2.warpPerspective(
        source_image,
        homography,
        (width, height),
    )

    return registered