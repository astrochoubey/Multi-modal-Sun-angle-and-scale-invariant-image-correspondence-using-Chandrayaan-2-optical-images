from typing import Any, Tuple, List, Optional
import cv2
import numpy as np


def draw_matches(
    source_image,
    source_keypoints,
    reference_image,
    reference_keypoints,
    matches,
    inlier_mask=None,
):
    """
    Draw feature correspondences between source and reference images.
    """

    if inlier_mask is not None:
        matches_to_draw = [
            match
            for match, is_inlier
            in zip(matches, inlier_mask)
            if is_inlier
        ]
    else:
        matches_to_draw = matches

    return cv2.drawMatches(
        source_image,
        source_keypoints,
        reference_image,
        reference_keypoints,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_registration_checkerboard(
    image1: np.ndarray,
    image2: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """
    Generate an alternating checkerboard pattern of two registered images
    for visual alignment and seam inspection.
    """
    h = min(image1.shape[0], image2.shape[0])
    w = min(image1.shape[1], image2.shape[1])

    img1 = image1[:h, :w]
    img2 = image2[:h, :w]

    # Convert to BGR if single channel
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    checkerboard = np.zeros_like(img1)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            r = (y // block_size) % 2
            c = (x // block_size) % 2
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)

            if (r + c) % 2 == 0:
                checkerboard[y:y_end, x:x_end] = img1[y:y_end, x:x_end]
            else:
                checkerboard[y:y_end, x:x_end] = img2[y:y_end, x:x_end]

    return checkerboard


def draw_registration_blend(
    image1: np.ndarray,
    image2: np.ndarray,
    alpha: float = 0.5,
    false_color: bool = True,
) -> np.ndarray:
    """
    Generate an overlay blend of two registered images.
    If false_color is True, displays reference in Cyan (Green+Blue) and
    registered source in Red: aligned terrain appears monochrome/gray,
    while registration errors appear with strong red or cyan color fringes.
    """
    h = min(image1.shape[0], image2.shape[0])
    w = min(image1.shape[1], image2.shape[1])

    img1 = image1[:h, :w]
    img2 = image2[:h, :w]

    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    if false_color:
        # False color anaglyph: Red channel = image2 (registered), Green+Blue = image1 (reference)
        # Perfectly registered areas are gray/white. Misaligned areas have cyan/red fringing.
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[..., 2] = img2_gray  # Red
        bgr[..., 1] = img1_gray  # Green
        bgr[..., 0] = img1_gray  # Blue
        return bgr
    else:
        return cv2.addWeighted(
            cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR),
            1.0 - alpha,
            cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR),
            alpha,
            0.0,
        )


def draw_grid_distribution_overlay(
    image: np.ndarray,
    points_or_keypoints: Any,
    grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Draw a spatial distribution grid over an image indicating keypoint density per cell.
    """
    h, w = image.shape[:2]
    if len(image.shape) == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    gx, gy = grid_size
    gw = w / float(gx)
    gh = h / float(gy)

    # Count points
    cell_counts = np.zeros((gy, gx), dtype=np.int32)
    pts_list = []
    for kp in points_or_keypoints:
        if hasattr(kp, "pt"):
            pt = kp.pt
        elif isinstance(kp, (list, tuple, np.ndarray)):
            pt = (float(kp[0]), float(kp[1]))
        else:
            continue
        pts_list.append(pt)
        cx = min(int(pt[0] / gw), gx - 1)
        cy = min(int(pt[1] / gh), gy - 1)
        cell_counts[cy, cx] += 1

    # Draw grid lines
    for x in range(1, gx):
        px = int(x * gw)
        cv2.line(canvas, (px, 0), (px, h), (70, 70, 70), 1)
    for y in range(1, gy):
        py = int(y * gh)
        cv2.line(canvas, (0, py), (w, py), (70, 70, 70), 1)

    # Draw keypoints as small green dots
    for pt in pts_list:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)

    # Overlay density count in each cell
    for r in range(gy):
        for c in range(gx):
            count = cell_counts[r, c]
            tx = int(c * gw + 4)
            ty = int((r + 1) * gh - 6)
            color = (0, 255, 255) if count > 0 else (100, 100, 100)
            cv2.putText(
                canvas,
                str(count),
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )

    return canvas
    