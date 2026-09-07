from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the image.

    Returns
    -------
    np.ndarray
        Image loaded in BGR format.

    Raises
    ------
    FileNotFoundError
        If the image cannot be loaded.
    """

    path = Path(path)

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image