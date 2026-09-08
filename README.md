# Lunar Image Registration & Correspondence Engine

### Multi-modal, Sun-Angle, and Scale-Invariant Image Registration for Chandrayaan-2 Planetary Observations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-38%2F38%20Passing-brightgreen.svg)]()
[![SIH-2026](https://img.shields.io/badge/SIH-Problem%2026166-purple.svg)]()

---

## 1. Executive Summary & Problem Statement

**Smart India Hackathon 2026 — Problem Statement ID 26166**  
**Title:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)  
**Organization:** Indian Space Research Organisation (ISRO)  

Planetary surface observations from orbit are acquired under radically varying observation geometries:
- **Solar Illumination & Sun Angles:** Solar elevation and azimuth angles change continuously, casting dynamic shadows, reversing shadow-lit rim edges, and washing out planar topography.
- **Multi-Sensor Spatial Scale:** Ground Sample Distance (GSD) ranges from sub-meter (Chandrayaan-2 OHRC: $\sim 0.25\text{ m/px}$) to regional stereo (TMC-2: $\sim 5.0\text{ m/px}$) and spectral mapping (IIRS: $\sim 80\text{ m/px}$).
- **Rugged 3D Topography:** Deep impact craters, towering central peaks, and steep crater walls violate the 2D planar assumption required by standard single-homography projective models.

This repository provides an **adaptive, lunar-aware registration system** that upgrades classical computer vision with:
1. **Terrain-Structure Representations** (Gradients, CLAHE, Local Contrast, Phase Congruency Proxy) that remain stable when raw pixel brightness inverts.
2. **Pre-Registration Pair Characterization** to quantify texture, illumination divergence, scale ratios, and topographic visual roughness prior to feature detection.
3. **Parsimonious Geometric Model Selection** that rigorously tests Translation (2 DOF), Similarity (4 DOF), Affine (6 DOF), and Homography (8 DOF), selecting the simplest model that explains correspondences without overfitting.
4. **Piecewise Local Homography with Distance-Feathered Warping** to register complex 3D relief without tearing or seam artifacts.
5. **Spatial Match Regularization** that evaluates grid occupancy and Gini concentration to prevent correspondence clustering on a single high-contrast crater rim.
6. **Explainable Composite Confidence Scoring** ($S_{conf} \in [0.0, 1.0]$) providing space mission operators with transparent verification metrics.
7. **Controlled Synthetic Benchmark Suite** with procedural crater generation and known ground-truth transforms to measure absolute sub-pixel error.

---

## 2. Theoretical Primer: Core Computer Vision Concepts

To make this codebase completely transparent and accessible to undergraduate researchers, engineers, and judges, this section breaks down the foundational math behind every pipeline stage.

### 2.1 Feature Detection & SIFT Scale-Space
The **Scale-Invariant Feature Transform (SIFT)** detects blobs and corner-like structures that persist across different scale levels:
1. **Scale-Space Construction:** The input image $I(x, y)$ is repeatedly convolved with variable-scale Gaussian kernels $G(x, y, \sigma)$:
   $$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$
2. **Difference-of-Gaussians (DoG):** Scale-space extrema are identified by subtracting adjacent octaves:
   $$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)$$
3. **Descriptor Vector:** Each keypoint is assigned a canonical orientation based on local image gradient directions, followed by computing an 8-bin orientation histogram over a $4 \times 4$ grid of sub-patches, resulting in a **128-dimensional invariant descriptor vector**.

### 2.2 Correspondence Matching & Lowe's Ratio Test
For each descriptor $d_{src}$ in the source image, Euclidean distance is measured against all descriptors in the reference image. To eliminate ambiguous matches occurring in repetitive terrain:
$$\text{Ratio} = \frac{\|d_{src} - d_{ref, 1}\|_2}{\|d_{src} - d_{ref, 2}\|_2} < \tau \quad (\text{typically } \tau = 0.75)$$
If the closest match $d_{ref, 1}$ is not distinctively closer than the second-closest match $d_{ref, 2}$, the match is discarded as ambiguous.

### 2.3 Outlier Rejection via RANSAC
Even after ratio testing, false matches occur due to sensor noise and terrain repetition. **Random Sample Consensus (RANSAC)** iteratively identifies the largest geometrically consistent consensus set:
1. Randomly sample the minimum points required to parameterize the geometric model (e.g. 4 points for Homography).
2. Fit candidate matrix $H$.
3. Compute symmetric transfer error for all correspondences: $e_i = \|p_i' - H p_i\|$.
4. Count inliers satisfying $e_i \le \text{threshold}$.
5. Re-estimate $H$ over the maximal consensus inlier set using Singular Value Decomposition (SVD).

### 2.4 Transformation Models & Parsimony
Planetary scenes do not always warrant an 8-DOF Homography:
- **Translation (2 DOF):** Rigid shift $[t_x, t_y]$. Ideal for narrow-baseline, nadir-pointing orbital frames with negligible rotation.
- **Similarity (4 DOF):** Translation + rotation $\theta$ + isotropic scale $s$. Preserves angles and aspect ratios.
- **Affine (6 DOF):** Similarity + anisotropic scaling + shear. Models non-orthogonal sensor pushbroom geometries.
- **Homography (8 DOF):** Planar perspective projection. Models arbitrary viewing angles of a planar surface.

**The Parsimony Principle:** *A higher-degree-of-freedom model is only accepted if its reprojection RMSE decreases by at least $15\%$ without a collapse in the inlier consensus set.* Overfitting an 8-DOF model on planar nadir imagery introduces unconstrained corner warping.

---

## 3. Why Classical Methods Struggle on Lunar Imagery

| Challenge | Physical Cause | Impact on Classical SIFT | Adaptive Solution |
| :--- | :--- | :--- | :--- |
| **Sun-Angle Variation** | Solar elevation and azimuth shift; crater shadows lengthen or reverse. | Raw intensity values invert across crater rims; SIFT gradient orientations flip by $180^\circ$, causing matching to fail. | **Structural representations** (Sobel gradient magnitude, local contrast, CLAHE) preserve slope boundaries regardless of shadow direction. |
| **Scale Discrepancy** | High-res framing camera (OHRC $\sim 0.25$ m) vs medium-res stereo (TMC-2 $\sim 5$ m). | Octave pyramids fail to overlap when scale jump exceeds $4\times$. | **Metadata-guided scale downsampling** & spectral frequency ratio proxy. |
| **Topographic Relief** | Craters and mountain massifs have significant 3D depth relative to orbit height. | Violates planar homography; single $3 \times 3$ matrix yields high residuals at crater rims. | **Piecewise local homography** with distance-feathered grid blending. |
| **Crater Rim Clustering** | High-contrast shadow boundaries on a single large crater dominate feature detection. | $90\%$ of correspondences fall in one quadrant; geometric fit is unstable elsewhere. | **Spatial match bucketing** and Gini concentration penalty. |

---

## 4. End-to-End System Architecture

```text
               Source Image (OHRC / TMC-2)              Reference Image (LROC / TMC-2)
                          │                                           │
                          ▼                                           ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │               1. Pair Characterization Engine                    │
              │  - Texture Variance & Entropy       - Bhattacharyya Distance      │
              │  - Spectral High-Frequency Ratio    - Visual Relief Proxy         │
              └─────────────────────────────────┬─────────────────────────────────┘
                                                │
                                                ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │               2. Adaptive Strategy Selector                       │
              │  Rule-based recommendation of representation, feature budget,     │
              │  matcher ratio threshold, and geometric complexity model.         │
              └─────────────────────────────────┬─────────────────────────────────┘
                                                │
                                                ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │               3. Terrain Structure Representation                 │
              │  Transform raw pixels -> CLAHE / Gradient / Local Contrast        │
              └─────────────────────────────────┬─────────────────────────────────┘
                                                │
                                                ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │               4. Feature Extraction & Matching                    │
              │  - SIFT with adaptive feature budget & contrast thresholds        │
              │  - Lowe's ratio test matching                                     │
              │  - Spatial match bucketing (caps points per grid cell)            │
              └─────────────────────────────────┬─────────────────────────────────┘
                                                │
                                                ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │               5. Parsimonious Model Selection                     │
              │  Fits Translation (2) -> Similarity (4) -> Affine (6) -> Homo (8) │
              │  Selects simplest model explaining data within RMSE threshold     │
              └─────────────────────────────────┬─────────────────────────────────┘
                                                │
                                                ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │               6. Warping & Piecewise Refinement                   │
              │  - Global projective warp OR 3x3 local tiled piecewise blend      │
              └─────────────────────────────────┬─────────────────────────────────┘
                                                │
                                                ▼
              ┌───────────────────────────────────────────────────────────────────┐
              │               7. Verification & Confidence Scoring                │
              │  Composite Score S_conf in [0, 1] + Checkerboard + Anaglyph Blend │
              └───────────────────────────────────────────────────────────────────┘
```

---

## 5. Installation & Setup

### Requirements
- Python 3.10, 3.11, 3.12, or 3.13
- OpenCV (`opencv-python` / `opencv-contrib-python`)
- NumPy
- PyYAML
- pytest (for automated tests)

### Clone & Install
```bash
# Clone the repository
git clone https://github.com/astrochoubey/Multi-modal-Sun-angle-and-scale-invariant-image-correspondence-using-Chandrayaan-2-optical-images.git
cd Multi-modal-Sun-angle-and-scale-invariant-image-correspondence-using-Chandrayaan-2-optical-images

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 6. CLI Command Guide

The toolkit provides 5 unified subcommands through `python -m lunar_registration`:

### 1. `register`: Classical Baseline Registration
Runs Atharv386's classical SIFT + RANSAC homography pipeline.
```bash
python -m lunar_registration register \
    --source data/raw/source.png \
    --reference data/raw/reference.png \
    --output-dir outputs/baseline
```
*Outputs:* `outputs/baseline/registered/registered.png`, `matches/matches.png`, `metrics/results.json`.

---

### 2. `adaptive`: Full Adaptive Registration Pipeline
Runs pair diagnostics, selects optimal representations, applies parsimonious model selection, runs piecewise warping if rugged terrain is detected, and scores registration confidence.
```bash
python -m lunar_registration adaptive \
    --source data/raw/source.png \
    --reference data/raw/reference.png \
    --output-dir outputs/adaptive_run
```
*Optional Flags:*
- `--gsd-source 0.5 --gsd-reference 5.0`: Ingest known Ground Sample Distance metadata.
- `--force-model similarity`: Force a specific geometric model (`translation`, `similarity`, `affine`, `homography`).
- `--no-piecewise`: Disable tiled local homography refinement.
- `--spatial-filter`: Enforce spatial match bucketing to cap crater-rim concentration.

*Terminal Output Example:*
```text
=================================================================
Adaptive Lunar Registration Pipeline
=================================================================
Source:    data/raw/source.png
Reference: data/raw/reference.png

-----------------------------------------------------------------
1. Pair Diagnostics:
  - Source Texture:       medium_texture (variance=910.0)
  - Reference Texture:    texture_rich (variance=1802.5)
  - Illumination Shift:   benign_illumination (Bhattacharyya dist=0.140)
  - Relief Visual Proxy:  high_relief_proxy (mean edge density=0.108)

2. Adaptive Strategy Selected:
  - Representation:       clahe
  - SIFT Feature Budget:  5000 (contrast thresh=0.03)
  - Lowe Ratio Cutoff:    0.75
  - Strategy Rationale:   Benign illumination: standard CLAHE contrast conditioning selected. | Spectral high-frequency disparity (0.10): adjusting scale pyramid. | High terrain relief proxy (rugged crater rims/massifs): selecting Homography with piecewise grid refinement contingency.

3. Geometric Model Comparison:
  Model          DOF   Inliers   Inlier Ratio   RMSE (px) 
  ------------------------------------------------------
  translation    2     35        0.673          2.136     
  similarity     4     34        0.654          1.390     
  affine         6     34        0.654          1.133     
  homography     8     40        0.769          1.312      [SELECTED]

4. Registration Outcome:
  - Status:               SUCCESS
  - Model Used:           HOMOGRAPHY
  - Inlier Matches:       40 / 52 (76.9%)
  - Reprojection RMSE:    1.3118 pixels
  - Piecewise Warp Used:  True

5. Explainable Confidence Assessment:
  - Score:                0.8015 / 1.0000
  - Classification:       HIGH_CONFIDENCE
  - Trustworthy:          True
  - Rationale:            Confidence HIGH_CONFIDENCE (0.80) evaluated from 40 inliers (76.9% ratio), 1.31px RMSE, and 40.6% spatial coverage.

6. Spatial Match Regularization:
  - Grid Coverage Ratio:  40.6% of cells occupied
  - Gini Concentration:   0.753 (0=uniform, 1=clustered)
```

*Generated Artifacts:*
- `registered.png`: Fully registered source image.
- `matches.png`: Side-by-side match correspondence lines.
- `checkerboard.png`: Alternating block visual seam alignment check.
- `blend.png`: False-color cyan/magenta anaglyph (aligned areas appear grey/monochrome; errors appear as color fringes).
- `spatial_distribution.png`: Grid cell match density map.
- `registration_report.json`: Full machine-readable audit report.

---

### 3. `characterize`: Image Pair Pre-Diagnostics
Computes quantitative image metrics prior to running any heavy registration algorithms.
```bash
python -m lunar_registration characterize \
    --source data/raw/source.png \
    --reference data/raw/reference.png
```

---

### 4. `benchmark`: Multi-Representation Benchmark
Evaluates all 7 representation filters on the same image pair to see which produces the highest inlier count and lowest RMSE.
```bash
python -m lunar_registration benchmark \
    --source data/raw/source.png \
    --reference data/raw/reference.png
```

---

### 5. `synthetic`: Controlled Ground-Truth Evaluation Suite
Generates simulated lunar crater surfaces with known geometric and radiometric distortions, evaluating corner transfer errors against absolute ground truth.
```bash
python -m lunar_registration synthetic --size 512
```

---

## 7. Five Suggested Reproducible Experiments

To explore the behavior of the registration pipeline, run these 5 standalone experiments:

### Experiment 1: Measuring Absolute Sub-Pixel Accuracy on Pure Translation
**Goal:** Verify whether classical vs adaptive estimation recovers integer and fractional pixel shifts accurately against ground truth.
```bash
python -m lunar_registration synthetic --size 512
```
*Expected Finding:* Translation and Similarity models achieve $<0.05$ px corner transfer error on pure synthetic rigid shifts.

### Experiment 2: Stress-Testing Keystone Tilt (Perspective Homography)
**Goal:** Test how well an 8-DOF Homography handles out-of-plane spacecraft viewing tilts.
```bash
python -m lunar_registration adaptive \
    --source data/raw/source.png \
    --reference data/raw/reference.png \
    --force-model homography \
    --output-dir outputs/exp2_homo
```
*Expected Finding:* Compare RMSE against `--force-model affine`. If the scene is planar, affine and homography show comparable RMSE, but homography may slightly deform unconstrained borders.

### Experiment 3: Extreme Sun Azimuth Shift (Why Gradients Win)
**Goal:** Observe what happens when raw pixel intensity values invert due to a $90^\circ+$ solar azimuth rotation.
```bash
python -m lunar_registration benchmark \
    --source data/raw/source.png \
    --reference data/raw/reference.png
```
*Expected Finding:* On severe shadow shifts, raw grayscale matching drops inliers sharply or fails, whereas gradient magnitude and local contrast representations maintain stable edge correspondences.

### Experiment 4: Spatial Match Regularization vs Crater Rim Clustering
**Goal:** Compare spatial distribution metrics with and without match bucketing.
```bash
# Without spatial filter (unconstrained)
python -m lunar_registration adaptive \
    --source data/raw/source.png \
    --reference data/raw/reference.png \
    --output-dir outputs/exp4_unconstrained

# With spatial filter (capped per grid cell)
python -m lunar_registration adaptive \
    --source data/raw/source.png \
    --reference data/raw/reference.png \
    --spatial-filter \
    --output-dir outputs/exp4_regularized
```
*Expected Finding:* The regularized run lowers the Gini concentration coefficient from $\sim 0.75$ to $<0.50$, distributing tie points evenly across the lunar mare rather than clustering on a single crater.

### Experiment 5: Model Parsimony vs Overfitting
**Goal:** Validate why the parsimony principle protects against unconstrained degree-of-freedom expansion.
```bash
python -m lunar_registration adaptive \
    --source data/raw/source.png \
    --reference data/raw/reference.png \
    --force-model translation \
    --output-dir outputs/exp5_trans
```
*Expected Finding:* Compare `registration_report.json` between Translation and Homography. Notice that Translation has 2 DOF with slightly higher RMSE ($2.1$ px vs $1.3$ px), but preserves scale and orthogonality perfectly.

---

## 8. Verification & Test Suite

All algorithms are covered by unit and integration test suites:
```bash
# Run the entire test suite
pytest tests/ -v
```

**Test Coverage Summary:**
- `test_representations.py`: Verifies all 7 representations (raw, clahe, gradient, local contrast, laplacian, edges, phase congruency) maintain shapes, dtypes, and range $[0, 255]$.
- `test_pair_characterization.py`: Tests texture variance, Shannon entropy, Bhattacharyya distance, and spectral scale ratio.
- `test_model_selection.py`: Verifies Translation, Similarity, Affine, Homography fitting, and parsimonious model ranking.
- `test_piecewise.py`: Verifies tiled local homography fitting and feather-blended perspective accumulation.
- `test_spatial_distribution.py`: Verifies grid coverage ratio, Gini inequality calculation, and spatial match bucketing.
- `test_confidence.py`: Verifies explainable confidence score bounds $[0.0, 1.0]$ and classification logic.
- `test_synthetic_suite.py`: Verifies crater generation, Lambertian shading, and ground-truth transform error calculation.
- `test_adaptive_pipeline.py`: Tests end-to-end execution of `adaptive_register_images`.

---

## 9. Repository Structure

```text
lunar-image-registration/
│
├── configs/                 # Baseline YAML configurations (sift, superpoint, loftr)
├── data/
│   ├── manifests/           # pilot_manifest.csv (DEV-01 to DEV-04 flight pairs)
│   ├── raw/                 # Raw test imagery (source.png, reference.png)
│   └── test/                # Controlled synthetic evaluation data
│
├── docs/                    # Architectural and research documentation
│   ├── research/            # Dossier, Annotated Bibliography, Prior Work Matrix
│   ├── data/                # Pilot Data Audit Report & Acquisition Guides
│   ├── dataset.md           # Sensor specifications (OHRC, TMC-2, IIRS)
│   ├── methodology.md       # Algorithmic formulation & math
│   └── experiments.md       # Experimental protocols & repeatability curves
│
├── notebooks/               # Jupyter research notebooks (01 to 07)
│
├── src/lunar_registration/  # Core Python Package
│   ├── __main__.py          # Entry point (python -m lunar_registration)
│   ├── cli.py               # Unified CLI (register, adaptive, characterize, benchmark, synthetic)
│   ├── adaptive/            # Interpretable strategy selection engine
│   ├── analysis/            # Pre-registration pair characterization
│   ├── evaluation/          # Metrics, confidence scoring, synthetic benchmark, visualizations
│   ├── features/            # SIFT feature detection & description
│   ├── geometry/            # Homography, model selection, piecewise local warping
│   ├── io/                  # Robust planetary raster loaders (.png, .tif, .img)
│   ├── matching/            # Lowe ratio test matcher & spatial distribution regularization
│   ├── preprocessing/       # Radiometric normalization & 7 structural representations
│   ├── registration/        # Classical and adaptive image warpers
│   └── utils/               # Coordinate helpers & array utilities
│
├── scripts/                 # Ingestion & data setup utilities
├── tests/                   # 38 passing unit tests
├── requirements.txt         # Project dependencies
├── pyproject.toml           # Build system configuration
└── README.md                # This documentation
```

---

## 10. Authors & License

- **Repository Maintainer:** Prachi Choubey ([@astrochoubey](https://github.com/astrochoubey))
- **Baseline Collaboration:** Atharv386
- **License:** [MIT License](LICENSE)
