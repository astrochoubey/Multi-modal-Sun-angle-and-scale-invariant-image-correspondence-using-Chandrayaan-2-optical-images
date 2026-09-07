# Lunar Image Registration

### Multi-modal, Sun-angle and Scale-invariant Image Correspondence for Chandrayaan-2 Optical Imagery

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)](#project-status)

## Overview

Lunar Image Registration is a research-oriented software system for automatically finding correspondences between lunar images acquired under different imaging conditions and registering them into a common coordinate system.

The project is designed around **Chandrayaan-2 optical observations**, including:

- Orbiter High Resolution Camera (OHRC)
- Terrain Mapping Camera-2 (TMC-2)
- Imaging Infrared Spectrometer (IIRS)

and lunar reference imagery such as:

- Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC)
- SELENE/Kaguya imagery

The system addresses three major challenges in lunar image correspondence:

1. **Illumination variation** caused by changes in solar azimuth and elevation.
2. **Viewpoint variation** caused by different camera positions and orientations.
3. **Scale variation** caused by different spacecraft altitudes and sensor resolutions.

The ultimate objective is to produce accurate, spatially distributed, potentially sub-pixel correspondences together with registered imagery and quantitative evaluation metrics.

---

## Problem Statement

**SIH Problem Statement ID:** 26166

**Title:** Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)

**Organization:** Indian Space Research Organisation (ISRO)

**Category:** Software

**Theme:** Space Technology

---

## Objectives

The project aims to develop a generic software pipeline capable of:

- Detecting reliable features in lunar imagery.
- Finding correspondences between images acquired by different sensors or missions.
- Handling changes in illumination and Sun angle.
- Handling scale differences between datasets.
- Handling viewpoint and geometric distortions.
- Rejecting incorrect correspondences.
- Maintaining a uniform spatial distribution of match points.
- Estimating geometric transformations robustly.
- Refining correspondences toward sub-pixel accuracy.
- Producing registered images.
- Providing quantitative registration metrics.

---

## System Architecture

```text
              Source Image                    Reference Image
                   │                                │
                   ▼                                ▼
             Preprocessing                   Preprocessing
                   │                                │
                   └──────────────┬─────────────────┘
                                  ▼
                         Feature Extraction
                                  │
                                  ▼
                             Matching
                                  │
                                  ▼
                         Outlier Rejection
                                  │
                                  ▼
                              RANSAC
                                  │
                                  ▼
                       Transformation Model
                                  │
                                  ▼
                        Sub-pixel Refinement
                                  │
                                  ▼
                            Registration
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
             Registered Image              Match Points
                   │                             │
                   └──────────────┬──────────────┘
                                  ▼
                              Evaluation
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                   RMSE      Inlier Ratio    Match Count
```

---

## Methodology

### 1. Preprocessing

The input imagery is normalized before feature extraction.

Potential preprocessing operations include:

- Grayscale conversion
- Intensity normalization
- Histogram equalization
- CLAHE
- Gamma correction
- Gradient-based representations
- Noise reduction
- Resolution normalization

The preprocessing stage will be evaluated experimentally to determine which representations provide the most reliable cross-illumination correspondence.

---

### 2. Feature Extraction

The system is designed to support multiple feature extraction approaches.

#### Classical Baselines

- SIFT
- ORB

#### Learned Features

- SuperPoint
- Other learned local feature methods

#### Dense Correspondence

- LoFTR

The classical methods provide reproducible baselines against which learned approaches can be evaluated.

---

### 3. Feature Matching

Detected features are matched between the source and reference images using descriptor-based or learned matching techniques.

The pipeline may employ:

- Nearest-neighbor matching
- Ratio testing
- Mutual/cross-check matching
- Descriptor-distance filtering
- Learned correspondence confidence

---

### 4. Outlier Rejection

Raw correspondences contain incorrect matches.

Robust geometric estimation is therefore applied using methods such as:

- RANSAC
- MAGSAC-style robust estimation
- Reprojection-error filtering

The resulting inlier set is used to estimate the final transformation.

---

### 5. Spatially Distributed Correspondences

A major design requirement is to avoid concentrating all correspondences in a small portion of the image.

The image can therefore be divided into spatial cells:

```text
┌─────┬─────┬─────┬─────┐
│     │  ●  │     │ ●   │
├─────┼─────┼─────┼─────┤
│ ●   │     │ ●   │     │
├─────┼─────┼─────┼─────┤
│     │ ●   │     │ ●   │
└─────┴─────┴─────┴─────┘
```

A controlled number of high-confidence correspondences can be selected from each region.

This improves geometric stability and ensures that the transformation is supported across the image rather than by a single localized feature cluster.

---

### 6. Geometric Registration

Depending on the imaging geometry, the system can estimate:

- Translation
- Similarity transformation
- Affine transformation
- Homography

The source image is then warped into the coordinate system of the reference image.

---

### 7. Sub-pixel Refinement

After coarse correspondence estimation, local optimization can refine feature locations beyond integer-pixel precision.

Conceptually:

```text
Detected point
     │
     ▼
Coarse correspondence
     │
     ▼
Local image patch
     │
     ▼
Optimization
     │
     ▼
Sub-pixel correspondence
```

This stage is important for meeting the desired high-accuracy registration objective.

---

## Evaluation

The system will report quantitative metrics including:

### RMSE

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|p_i-\hat{p}_i\right\|^2
}
$$

### Inlier Count

Number of correspondences satisfying the selected geometric error threshold.

### Inlier Ratio

$$
Inlier\ Ratio =
\frac{N_{inliers}}{N_{matches}}
$$

### Additional Metrics

The benchmark will also track:

- Total keypoints
- Raw matches
- Filtered matches
- Inlier matches
- Reprojection error
- Spatial coverage
- Registration error
- Processing time

---

## Project Structure

```text
lunar-image-registration/
│
├── configs/                 # Experiment configurations
├── data/
│   ├── raw/                 # Original imagery
│   ├── processed/           # Preprocessed data
│   ├── patches/             # Image patches
│   └── test/                # Test samples
│
├── notebooks/               # Research and experimentation
│
├── src/
│   └── lunar_registration/
│       ├── io/
│       ├── preprocessing/
│       ├── features/
│       ├── matching/
│       ├── geometry/
│       ├── registration/
│       ├── evaluation/
│       └── utils/
│
├── scripts/                 # Dataset and experiment scripts
├── tests/                   # Automated tests
├── outputs/
│   ├── matches/
│   ├── registered/
│   ├── metrics/
│   └── figures/
│
├── docs/                    # Technical documentation
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd lunar-image-registration
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Basic Usage

Run registration on an image pair:

```bash
python -m lunar_registration.cli register \
    --source path/to/source.tif \
    --reference path/to/reference.tif \
    --config configs/default.yaml
```

Run evaluation:

```bash
python -m lunar_registration.cli evaluate \
    --config configs/default.yaml
```

Run the complete benchmark:

```bash
python -m lunar_registration.cli benchmark
```

---

## Development Roadmap

### Phase 1 — Baseline

- [ ] Dataset ingestion
- [ ] Image preprocessing
- [ ] SIFT implementation
- [ ] Descriptor matching
- [ ] Ratio test
- [ ] RANSAC
- [ ] Homography estimation
- [ ] Image registration
- [ ] Match visualization
- [ ] RMSE calculation
- [ ] Inlier statistics

### Phase 2 — Robustness

- [ ] Illumination normalization
- [ ] Sun-angle robustness experiments
- [ ] Scale robustness experiments
- [ ] Viewpoint robustness experiments
- [ ] Spatially distributed matching
- [ ] Improved outlier rejection

### Phase 3 — Learned Correspondence

- [ ] SuperPoint
- [ ] Learned descriptor matching
- [ ] LoFTR
- [ ] Cross-sensor experiments
- [ ] Comparative benchmark

### Phase 4 — High-Precision Registration

- [ ] Sub-pixel refinement
- [ ] Local optimization
- [ ] Improved geometric models
- [ ] Accuracy analysis

### Phase 5 — Final System

- [ ] End-to-end pipeline
- [ ] Automated benchmarking
- [ ] Result visualization
- [ ] Reproducible experiments
- [ ] Documentation
- [ ] Software demonstration

---

## Experimental Design

The project will evaluate the effect of:

### Illumination

```text
Different Sun elevation
Different Sun azimuth
Different shadow conditions
Different surface brightness
```

### Scale

```text
High resolution
      ↕
Medium resolution
      ↕
Low resolution
```

### Viewpoint

```text
Translation
Rotation
Affine deformation
Perspective deformation
```

Each experiment will compare correspondence quality using consistent evaluation metrics.

---

## Expected Output

For every source/reference image pair, the system should produce:

```text
outputs/
├── registered/
│   └── registered_image.tif
│
├── matches/
│   └── correspondences.png
│
└── metrics/
    └── results.json
```

Example result:

```json
{
  "keypoints_source": 1243,
  "keypoints_reference": 1587,
  "raw_matches": 842,
  "inliers": 613,
  "inlier_ratio": 0.728,
  "rmse": 0.84,
  "spatial_coverage": 0.91
}
```

The numerical values above are illustrative output fields, not experimental results.

---

## Data Sources

The problem statement identifies Chandrayaan-2 optical imagery including OHRC, TMC-2 and IIRS, with reference imagery from LRO NAC and SELENE.

Dataset access and availability will be incorporated once the official datasets are provided.

Potential official sources include:

- Chandrayaan-2 / Indian Space Science Data Centre
- LRO NAC
- SELENE/Kaguya

Dataset licensing, access restrictions and metadata requirements will be documented before distributing any data with the repository.

---

## Research Direction

The project is not intended to be merely an implementation of a single existing registration algorithm.

The research direction is to investigate how lunar-specific factors affect image correspondence and develop a robust registration pipeline capable of handling:

$$
\boxed{
Illumination +
Scale +
Viewpoint +
Cross\text{-}modal\ differences
}
$$

while maintaining:

$$
\boxed{
High\ accuracy +
Spatially\ distributed\ correspondences +
Sub\text{-}pixel\ precision
}
$$

The final system will be evaluated against classical and learned correspondence baselines.

---

## Reproducibility

All experiments should be configuration-driven.

Each experiment should record:

```text
Dataset
Sensor
Preprocessing
Feature extractor
Matcher
RANSAC parameters
Transformation model
Refinement method
Metrics
Runtime
```

This allows experiments to be reproduced and compared systematically.

---

## License

This project is intended for research and educational development.

The final license will be selected after confirming the licensing requirements of the datasets, models and third-party libraries used by the project.

---

## Acknowledgements

Developed as part of **Smart India Hackathon 2026** for the Indian Space Research Organisation (ISRO), Department of Space.

**Problem Statement:** 26166
