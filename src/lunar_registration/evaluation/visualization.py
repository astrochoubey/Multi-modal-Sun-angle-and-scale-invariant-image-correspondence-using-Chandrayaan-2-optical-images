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
    