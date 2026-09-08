"""
Local Web Server for Lunar Image Registration Studio.
Provides a FastAPI backend and interactive web interface to run, visualize,
and benchmark Chandrayaan-2 image correspondence under varying Sun angles and scales.
"""

import base64
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from lunar_registration.features.sift import detect_and_compute
from lunar_registration.geometry.homography import estimate_homography
from lunar_registration.matching.matcher import match_descriptors
from lunar_registration.preprocessing.preprocessing import apply_clahe, to_grayscale
from lunar_registration.registration.register import warp_image
from lunar_registration.evaluation.metrics import (
    calculate_inlier_ratio,
    calculate_rmse,
    calculate_spatial_coverage,
)
from lunar_registration.evaluation.visualization import draw_matches
from scripts.generate_sample_data import (
    generate_dataset,
    get_real_lunar_scenario,
    get_real_paths,
)

app = FastAPI(title="Lunar Image Registration Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSPACE_ROOT = Path(__file__).resolve().parent
WEB_DIR = WORKSPACE_ROOT / "web"
DATA_DIR = WORKSPACE_ROOT / "data"
OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"


def image_to_base64(img: np.ndarray, quality: int = 90) -> str:
    """Encode OpenCV BGR image as Base64 JPEG data URL."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def make_checkerboard(img1: np.ndarray, img2: np.ndarray, grid_size: int = 8) -> np.ndarray:
    """Combine two images into an alternating checkerboard pattern."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    checker = img1.copy()
    cell_h, cell_w = h // grid_size, w // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if (i + j) % 2 == 1:
                checker[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w] = (
                    img2_resized[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                )
    return checker


def make_difference_heatmap(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Generate colorized absolute difference heatmap between reference and registered."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    mask = (g2 > 0).astype(np.uint8) * 255
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
    return heatmap


def run_pipeline(
    source_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    n_features: int = 5000,
    ratio_threshold: float = 0.75,
    reprojection_threshold: float = 5.0,
    clip_limit: float = 2.0,
) -> dict:
    """Execute the full Lunar Image Registration pipeline."""
    start_time = time.time()

    # 1. Preprocessing
    src_gray = to_grayscale(source_bgr)
    ref_gray = to_grayscale(reference_bgr)
    src_prep = apply_clahe(src_gray, clip_limit=clip_limit)
    ref_prep = apply_clahe(ref_gray, clip_limit=clip_limit)

    # 2. SIFT feature detection
    src_kp, src_desc = detect_and_compute(src_prep, n_features=n_features)
    ref_kp, ref_desc = detect_and_compute(ref_prep, n_features=n_features)

    if src_desc is None or ref_desc is None or len(src_kp) == 0 or len(ref_kp) == 0:
        raise ValueError("Could not detect sufficient features in one of the images.")

    # 3. Matching
    matches = match_descriptors(src_desc, ref_desc, ratio_threshold=ratio_threshold)
    if len(matches) < 4:
        raise ValueError(f"Insufficient matches ({len(matches)} found, minimum 4 required).")

    # 4. Homography with RANSAC
    homography, inlier_mask = estimate_homography(
        src_kp,
        ref_kp,
        matches,
        reprojection_threshold=reprojection_threshold,
    )

    inliers_count = int(inlier_mask.sum())
    inlier_ratio = calculate_inlier_ratio(inlier_mask)

    # 5. Warping
    registered_bgr = warp_image(source_bgr, homography, reference_bgr.shape)

    # 6. Evaluation metrics
    inlier_matches = [m for m, is_in in zip(matches, inlier_mask) if is_in]
    rmse = calculate_rmse(src_kp, ref_kp, inlier_matches, homography)
    spatial_coverage = calculate_spatial_coverage(ref_kp, inlier_matches, reference_bgr.shape)

    # 7. Visualizations
    match_vis = draw_matches(source_bgr, src_kp, reference_bgr, ref_kp, matches, inlier_mask)
    checkerboard = make_checkerboard(reference_bgr, registered_bgr)
    diff_map = make_difference_heatmap(reference_bgr, registered_bgr)

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "metrics": {
            "source_keypoints": len(src_kp),
            "reference_keypoints": len(ref_kp),
            "good_matches": len(matches),
            "inliers": inliers_count,
            "inlier_ratio": round(inlier_ratio * 100, 2),
            "rmse_pixels": round(rmse, 4),
            "spatial_coverage": round(spatial_coverage * 100, 2),
            "latency_ms": elapsed_ms,
            "homography": homography.tolist(),
        },
        "images": {
            "reference": image_to_base64(reference_bgr),
            "source": image_to_base64(source_bgr),
            "registered": image_to_base64(registered_bgr),
            "matches": image_to_base64(match_vis),
            "checkerboard": image_to_base64(checkerboard),
            "difference": image_to_base64(diff_map),
        },
    }


@app.get("/api/presets")
def get_presets():
    """List available real lunar imagery presets."""
    return {
        "presets": [
            {
                "id": "primary",
                "name": "Chandrayaan-2 Real Observation Pair",
                "description": "Full-resolution real optical lunar observation pair (real_source.png & real_reference.png).",
            },
            {
                "id": "illumination",
                "name": "Real Data: Sun-Angle Illumination Invariance",
                "description": "Real lunar imagery evaluated under solar elevation lighting gradients and surface shadows.",
            },
            {
                "id": "scale",
                "name": "Real Data: Cross-Sensor Scale Invariance",
                "description": "Simulates multi-sensor ground sampling distance (GSD) mismatch between OHRC and TMC-2.",
            },
            {
                "id": "crater_crop",
                "name": "Real Data: High-Resolution Crater Basin",
                "description": "Detailed focus on real lunar crater basin rim structures and ejecta morphology.",
            },
        ]
    }


@app.post("/api/register")
async def register_endpoint(
    preset_id: Optional[str] = Form("primary"),
    n_features: int = Form(6000),
    ratio_threshold: float = Form(0.75),
    reprojection_threshold: float = Form(5.0),
    clip_limit: float = Form(2.0),
    source_file: Optional[UploadFile] = File(None),
    reference_file: Optional[UploadFile] = File(None),
):
    """Execute registration on either an uploaded pair or a real lunar observation preset."""
    try:
        # Check if custom files uploaded
        if source_file and reference_file:
            src_bytes = await source_file.read()
            ref_bytes = await reference_file.read()
            src_arr = np.frombuffer(src_bytes, np.uint8)
            ref_arr = np.frombuffer(ref_bytes, np.uint8)
            source_bgr = cv2.imdecode(src_arr, cv2.IMREAD_COLOR)
            reference_bgr = cv2.imdecode(ref_arr, cv2.IMREAD_COLOR)
            if source_bgr is None or reference_bgr is None:
                return JSONResponse(status_code=400, content={"error": "Failed to decode uploaded images."})
        else:
            # Map legacy or alias ids to real lunar scenarios
            scenario_map = {
                "real_data": "primary",
                "primary": "primary",
                "illumination": "illumination",
                "scale": "scale",
                "complex_crater": "crater_crop",
                "crater_crop": "crater_crop",
            }
            scenario_id = scenario_map.get(preset_id, "primary")
            source_bgr, reference_bgr = get_real_lunar_scenario(scenario_id, root=DATA_DIR)

        result = run_pipeline(
            source_bgr=source_bgr,
            reference_bgr=reference_bgr,
            n_features=n_features,
            ratio_threshold=ratio_threshold,
            reprojection_threshold=reprojection_threshold,
            clip_limit=clip_limit,
        )
        return JSONResponse(content={"success": True, **result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# Mount static assets if web directory exists
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def serve_home():
    """Serve the web app homepage."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Lunar Image Registration API is running. Web assets in progress."})


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Launch the local server."""
    print(f"\n=======================================================")
    print(f"  CHANDRAYAAN-2 LUNAR IMAGE REGISTRATION STUDIO")
    print(f"  Local Server: http://{host}:{port}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
