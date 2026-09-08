"""
Image and results serialization utilities.
"""

from pathlib import Path
import json
import cv2
import numpy as np


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save an image to disk, ensuring directory exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to write image to {path}")


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Save metrics dictionary to a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
