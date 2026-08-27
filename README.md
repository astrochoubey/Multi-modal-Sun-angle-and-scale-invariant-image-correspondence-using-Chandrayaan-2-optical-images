# Lunar Image Registration

A robust computer-vision framework for registering **Chandrayaan-2 optical lunar images** with corresponding **lunar reference images**. The system identifies reliable correspondence points between images acquired under different illumination, viewpoints, and spatial resolutions, and geometrically transforms the source image into the reference image's coordinate system.

## Overview

**Image registration** is the process of aligning two or more images of the same scene into a common coordinate system.

In this project:

- **Source / Moving Image:** Chandrayaan-2 acquired optical image that is geometrically transformed.
- **Reference / Fixed Image:** Lunar reference image used as the target coordinate system.

The primary objective is to develop a **generic software solution capable of producing accurate and spatially distributed correspondence points with sub-pixel registration accuracy**.

---

## Problem Statement

Registering lunar images is challenging because images of the same lunar region may be acquired:

- At different times
- From different camera viewpoints
- At different altitudes
- At different spatial resolutions
- Under significantly different illumination conditions

These differences make conventional feature matching unreliable and can introduce substantial geometric errors.

### Major Challenges

#### 1. Illumination Variation

Changes in solar azimuth and elevation alter the shadows, brightness, and appearance of lunar surface features.

A crater may therefore appear significantly different between two observations even though its physical location has not changed.

#### 2. Viewpoint Variation

Different camera positions and orientations introduce geometric distortions.

Surface features can appear:

- Translated
- Rotated
- Scaled
- Perspective-distorted

#### 3. Scale Variation

Lunar missions can acquire imagery from vastly different orbital altitudes and with different imaging resolutions.

Consequently, the same surface feature may occupy substantially different numbers of pixels in the two images.

#### 4. Incorrect Correspondences

Feature matching can produce false matches, particularly in repetitive or low-texture lunar terrain.

Robust outlier rejection is therefore essential.

#### 5. Sub-pixel Accuracy

The final registered product should achieve correspondence accuracy below one pixel wherever the image quality and available information permit.

#### 6. Spatial Distribution of Matches

A large number of matches concentrated in one small region is not sufficient.

The correspondence points should be distributed uniformly across the overlapping image area so that the estimated transformation remains stable.

---

## Objectives

The project aims to:

1. Automatically identify corresponding points between Chandrayaan-2 and reference lunar imagery.
2. Handle illumination, viewpoint, and scale variations.
3. Reject incorrect feature correspondences.
4. Estimate an appropriate geometric transformation.
5. Refine correspondence locations to achieve sub-pixel accuracy.
6. Maintain a uniform spatial distribution of reliable match points.
7. Generate a registered Chandrayaan-2 image.
8. Provide the corresponding match-point dataset.
9. Quantitatively evaluate registration quality.

---

## Proposed Pipeline

```text
                 ┌──────────────────────┐
                 │ Chandrayaan-2 Image  │
                 │   Source / Moving    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Preprocessing     │
                 │ Normalization/Noise │
                 │     Reduction        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Multi-scale Feature │
                 │     Extraction       │
                 └──────────┬───────────┘
                            │
                            │
        ┌───────────────────┘
        │
        ▼
┌──────────────────────┐
│ Reference Image      │
│    Fixed Image       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature Extraction   │
└──────────┬───────────┘
           │
           └──────────────┐
                          ▼
                 ┌──────────────────────┐
                 │ Feature / Descriptor │
                 │      Matching        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Match Filtering      │
                 │ Ratio + Mutual Test  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Robust Transformation│
                 │       RANSAC         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Sub-pixel Refinement │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Spatial Distribution │
                 │    / Grid Sampling   │
                 └──────────┬───────────┘
                            │
                ┌───────────┴────────────┐
                ▼                        ▼
       ┌─────────────────┐      ┌─────────────────┐
       │ Registered Image│      │ Correspondence  │
       │                 │      │     Points      │
       └────────┬────────┘      └────────┬────────┘
                │                        │
                └────────────┬───────────┘
                             ▼
                    ┌──────────────────┐
                    │    Evaluation    │
                    │ RMSE / Inliers / │
                    │  Inlier Ratio    │
                    └──────────────────┘
```

---

## Feature Matching

The framework can support multiple feature extraction and matching techniques.

### Initial Baseline

**SIFT (Scale-Invariant Feature Transform)** is recommended as the initial baseline because it provides robustness to:

- Scale changes
- Rotation
- Moderate viewpoint changes
- Local appearance variations

### Potential Advanced Methods

The framework can later evaluate:

- ORB
- SuperPoint
- LoFTR
- Other learned feature-matching architectures

This allows the project to quantitatively compare classical and deep-learning-based approaches.

---

## Geometric Registration

After obtaining candidate correspondences, robust geometric estimation is performed.

Depending on the characteristics of the image pair, possible transformation models include:

- Translation
- Affine transformation
- Homography
- Piecewise geometric transformation
- More advanced non-rigid models where required

RANSAC or another robust estimator is used to remove outlier correspondences.

The transformation can be represented generally as:

$$
x_r = Hx_s
$$

where:

- \(x_s\) = source-image coordinate
- \(x_r\) = reference-image coordinate
- \(H\) = estimated geometric transformation

The simplest model capable of accurately representing the observed distortion should be preferred.

---

## Sub-pixel Refinement

Initial feature matches generally provide pixel-level coordinates.

To improve registration accuracy, the selected correspondences can undergo local refinement using techniques such as:

- Lucas-Kanade optimization
- Local template matching
- Phase correlation
- Intensity-based optimization
- Local optical-flow refinement
- Least-squares optimization

The final objective is to minimize the residual correspondence error and achieve **sub-pixel accuracy**.

---

## Uniform Match Distribution

To prevent matches from clustering in a small region, the overlapping image area can be divided into a grid.

For example:

```text
┌──────┬──────┬──────┬──────┐
│  •   │      │  •   │      │
├──────┼──────┼──────┼──────┤
│      │  •   │      │  •   │
├──────┼──────┼──────┼──────┤
│  •   │      │  •   │      │
├──────┼──────┼──────┼──────┤
│      │  •   │      │  •   │
└──────┴──────┴──────┴──────┘
```

A fixed or adaptive number of high-quality correspondences can then be selected from each grid cell.

This provides:

- Better transformation stability
- Better coverage
- Reduced dependence on a single surface feature
- More reliable registration across the entire image

---

## Evaluation Metrics

The system should report multiple metrics rather than relying on a single accuracy measurement.

### Root Mean Square Error

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left[
(x_i-\hat{x}_i)^2+
(y_i-\hat{y}_i)^2
\right]
}
$$

### Inlier Count

Number of correspondence points that remain consistent with the estimated geometric transformation.

### Inlier Ratio

$$
Inlier\ Ratio =
\frac{N_{inliers}}{N_{matches}}
$$

### Additional Metrics

The framework can also report:

- Median reprojection error
- 95th-percentile reprojection error
- Maximum reprojection error
- Number of valid correspondence points
- Spatial coverage
- Registration success rate
- Processing time

---

## Expected Outputs

For every successfully registered image pair, the system should produce:

### 1. Registered Image

The Chandrayaan-2 source image transformed into the reference image's coordinate system.

### 2. Match Points

A structured file containing corresponding coordinates:

```text
source_x, source_y, reference_x, reference_y, error, confidence
```

### 3. Transformation Parameters

The estimated transformation matrix/model used for registration.

### 4. Evaluation Report

Example:

```text
Initial Matches       : 1842
Filtered Matches      : 967
RANSAC Inliers        : 812
Inlier Ratio          : 83.97%
RMSE                  : 0.42 px
Median Error          : 0.28 px
Spatial Coverage     : 91.4%
```

---

## Suggested Project Structure

```text
lunar-image-registration/
│
├── data/
│   ├── source/
│   └── reference/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── matching/
│   ├── registration/
│   ├── refinement/
│   ├── distribution/
│   └── evaluation/
│
├── outputs/
│   ├── registered/
│   ├── matches/
│   └── reports/
│
├── tests/
│
├── configs/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Technology Stack

The initial implementation can be developed using:

- **Python**
- OpenCV
- NumPy
- SciPy
- scikit-image
- Matplotlib
- PyTorch — for learned feature/matching models

Potential GPU acceleration can be introduced for deep-learning-based methods and large-scale image processing.

---

## Development Roadmap

### Phase 1 — Baseline

- Load source and reference images
- Preprocess images
- Implement SIFT feature extraction
- Implement descriptor matching
- Apply ratio-test filtering
- Estimate homography using RANSAC
- Generate registered image

### Phase 2 — Robustness

- Handle illumination variation
- Introduce image pyramids
- Improve outlier rejection
- Implement spatially distributed matching
- Add quantitative evaluation

### Phase 3 — Sub-pixel Registration

- Implement local correspondence refinement
- Calculate sub-pixel residuals
- Optimize transformation parameters
- Validate accuracy on multiple image pairs

### Phase 4 — Advanced Matching

Compare classical methods against learned approaches such as:

```text
SIFT
  ↓
SuperPoint
  ↓
LoFTR
```

Evaluate each method using the same dataset and metrics.

### Phase 5 — Production Pipeline

- Automated batch processing
- Configuration-based execution
- Visualization tools
- Structured output generation
- Evaluation reports
- Error handling
- Performance optimization

---

## Visualization

The system should provide visual diagnostics such as:

### Match Visualization

```text
SOURCE IMAGE                    REFERENCE IMAGE

     •───────────────•
       \             \
        •───────────────•
          \           /
           •─────────•
```

### Registration Overlay

The registered source image can be overlaid with the reference image to visually identify:

- Correct alignment
- Residual shifts
- Local geometric distortions
- Areas with poor registration

---

## Success Criteria

The proposed system will be considered successful if it can consistently:

- Automatically register Chandrayaan-2 optical images against reference imagery.
- Handle substantial illumination variation.
- Handle scale and viewpoint differences.
- Reject incorrect correspondences.
- Produce spatially distributed reliable match points.
- Achieve sub-pixel correspondence accuracy where feasible.
- Generate a reproducible quantitative evaluation report.
- Process multiple image pairs without manual intervention.

---

## Future Extensions

Potential future improvements include:

- Digital Elevation Model (DEM)-assisted registration
- Orthorectification
- Physics-informed illumination normalization
- Lunar terrain-aware feature extraction
- GPU acceleration
- Deep-learning-based correspondence estimation
- Uncertainty estimation for individual match points
- Automatic quality assessment
- Large-scale Chandrayaan-2 image catalogue registration

---

## Project Goal

The ultimate goal is to develop a **generic and extensible lunar image registration system** capable of reliably aligning Chandrayaan-2 optical imagery with lunar reference datasets despite differences in illumination, viewpoint, and scale.

The system should provide not only a registered image, but also **high-quality, spatially distributed correspondence points and quantitative evidence of registration accuracy**, making the resulting products suitable for downstream lunar mapping, analysis, and scientific applications.
