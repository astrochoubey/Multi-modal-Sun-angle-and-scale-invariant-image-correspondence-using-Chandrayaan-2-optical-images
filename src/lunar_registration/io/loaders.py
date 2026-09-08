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

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None and path.suffix.lower() in [".img", ".raw", ".bin"]:
        # Fallback for uncompressed planetary rasters (1024x1024 standard tile)
        file_bytes = path.stat().st_size
        num_pixels = 1024 * 1024
        if file_bytes >= num_pixels * 2:
            raw = np.fromfile(str(path), dtype=">u2")
            if len(raw) < num_pixels:
                raw = np.fromfile(str(path), dtype="<u2")
            image = raw[:num_pixels].reshape((1024, 1024))
        else:
            raw = np.fromfile(str(path), dtype=np.uint8)
            image = raw[:num_pixels].reshape((1024, 1024))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image