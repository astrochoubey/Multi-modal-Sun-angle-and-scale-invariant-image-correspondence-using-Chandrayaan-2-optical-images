import argparse
import json
from pathlib import Path

import cv2

from lunar_registration.io.loaders import load_image
from lunar_registration.preprocessing.preprocessing import preprocess
from lunar_registration.features.sift import detect_and_compute
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import (
    calculate_rmse,
    calculate_inlier_ratio,
)
from lunar_registration.evaluation.visualization import draw_matches


def register_images(source_path, reference_path):

    print("Loading images...")

    source = load_image(source_path)
    reference = load_image(reference_path)

    print("Preprocessing...")

    source_processed = preprocess(source)
    reference_processed = preprocess(reference)

    print("Detecting SIFT features...")

    source_keypoints, source_descriptors = detect_and_compute(
        source_processed
    )

    reference_keypoints, reference_descriptors = detect_and_compute(
        reference_processed
    )

    print(
        f"Source keypoints: {len(source_keypoints)}"
    )

    print(
        f"Reference keypoints: {len(reference_keypoints)}"
    )

    print("Matching descriptors...")

    matches = match_descriptors(
        source_descriptors,
        reference_descriptors,
    )

    print(
        f"Good matches: {len(matches)}"
    )

    if len(matches) < 4:
        raise RuntimeError(
            "Not enough matches for homography estimation."
        )

    print("Estimating homography with RANSAC...")

    homography, inlier_mask = estimate_homography(
        source_keypoints,
        reference_keypoints,
        matches,
    )

    inlier_count = int(inlier_mask.sum())

    inlier_ratio = calculate_inlier_ratio(
        inlier_mask
    )

    print(
        f"Inliers: {inlier_count}"
    )

    print(
        f"Inlier ratio: {inlier_ratio:.3f}"
    )

    print("Warping source image...")

    registered = warp_image(
        source,
        homography,
        reference.shape,
    )

    print("Calculating RMSE...")

    inlier_matches = [
        match
        for match, is_inlier
        in zip(matches, inlier_mask)
        if is_inlier
    ]

    rmse = calculate_rmse(
        source_keypoints,
        reference_keypoints,
        inlier_matches,
        homography,
    )

    print(
        f"RMSE: {rmse:.4f} pixels"
    )

    output_dir = Path("outputs")

    matches_dir = output_dir / "matches"
    registered_dir = output_dir / "registered"
    metrics_dir = output_dir / "metrics"

    matches_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    registered_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    match_visualization = draw_matches(
        source,
        source_keypoints,
        reference,
        reference_keypoints,
        matches,
        inlier_mask,
    )

    cv2.imwrite(
        str(matches_dir / "matches.png"),
        match_visualization,
    )

    cv2.imwrite(
        str(registered_dir / "registered.png"),
        registered,
    )

    metrics = {
        "source_keypoints": len(source_keypoints),
        "reference_keypoints": len(reference_keypoints),
        "good_matches": len(matches),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "rmse_pixels": rmse,
        "homography": homography.tolist(),
    }

    with open(
        metrics_dir / "results.json",
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nRegistration complete.")

    print(
        f"Registered image: "
        f"{registered_dir / 'registered.png'}"
    )

    print(
        f"Matches: "
        f"{matches_dir / 'matches.png'}"
    )

    print(
        f"Metrics: "
        f"{metrics_dir / 'results.json'}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Lunar Image Registration"
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    register_parser = subparsers.add_parser(
        "register"
    )

    register_parser.add_argument(
        "--source",
        required=True,
        help="Path to source image",
    )

    register_parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference image",
    )

    args = parser.parse_args()

    if args.command == "register":

        register_images(
            args.source,
            args.reference,
        )


if __name__ == "__main__":
    main()