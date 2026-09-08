# 🌕 Lunar Image Registration

### Adaptive, Illumination-Robust and Scale-Aware Image Correspondence for Chandrayaan-2 Optical Imagery

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)](#project-status)
[![Domain](https://img.shields.io/badge/Domain-Lunar%20Computer%20Vision-purple)](#research-direction)

> **Research principle:** Do not assume that two lunar images should
> have similar pixels. Instead, identify and match the **terrain
> structure** that remains useful when illumination, scale, viewpoint,
> and sensor characteristics change.

------------------------------------------------------------------------

## Overview

**Lunar Image Registration** is a research-oriented computer-vision
framework for finding reliable correspondences between lunar images and
aligning them into a common coordinate system.

The target use case is cross-sensor and cross-mission lunar imagery,
including Chandrayaan-2 observations such as:

-   **OHRC** --- Orbiter High Resolution Camera
-   **TMC-2** --- Terrain Mapping Camera-2
-   **IIRS** --- Imaging Infrared Spectrometer

and reference imagery such as:

-   **LRO NAC** --- Lunar Reconnaissance Orbiter Narrow Angle Camera
-   **SELENE/Kaguya** imagery

The SIH problem requires correspondence under changing **Sun angle,
scale, viewpoint, and sensor characteristics**, together with spatially
distributed matches and accurate registration.

Our project goes beyond a fixed `SIFT → RANSAC → Homography` pipeline.

The proposed system is **adaptive**:

1.  characterize the image pair,
2.  generate illumination-robust terrain representations,
3.  choose an appropriate correspondence strategy,
4.  estimate and validate the geometric model,
5.  fall back from global to local registration when the terrain
    requires it,
6.  refine the result,
7.  report quantitative confidence.

The SIH problem statement itself identifies illumination, viewpoint, and
scale as central challenges. fileciteturn1file0

------------------------------------------------------------------------

# 🎯 Problem Statement

**SIH Problem Statement ID:** 26166

**Title:** *Multi-modal, Sun angle and scale invariant image
correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)*

**Organization:** Indian Space Research Organisation (ISRO)

**Category:** Software

**Theme:** Space Technology

The problem is fundamentally an **image correspondence and registration
problem**:

> Given two observations of approximately the same lunar terrain, find
> reliable corresponding points and transform the moving/source image
> into the coordinate system of the fixed/reference image.

The difficulty is that the same terrain may look very different because:

-   the Sun illuminates it from a different direction,
-   shadows move,
-   spatial resolution changes,
-   the spacecraft viewpoint changes,
-   different sensors have different imaging characteristics,
-   terrain relief produces spatially varying geometric distortion.

------------------------------------------------------------------------

# 💡 Core Research Idea

Traditional registration often starts with:

``` text
Image A
   ↓
SIFT
   ↓
Match
   ↓
RANSAC
   ↓
Homography
```

Our research question is more specific to the Moon:

> **If the pixels change strongly but the underlying terrain structure
> remains related, can we make registration more robust by explicitly
> representing and selecting stable terrain structure?**

Instead of asking only:

> "Do these pixels look similar?"

we ask:

> **"Does the underlying terrain structure look similar?"**

This motivates a family of representations:

-   CLAHE / local contrast
-   gradient magnitude
-   edge maps
-   Laplacian structure
-   phase congruency
-   shadow-suppressed representations
-   multi-scale representations

These are **candidate representations**, not assumptions about which
method is universally best. The benchmark should determine which
representation is most stable for each imaging condition.

This direction is strongly motivated by planetary-registration
literature: lunar/planetary imagery is known to suffer from low
contrast, uneven illumination, shadow effects, weak textures, and
geometric changes. Earlier planetary work explicitly noted that methods
developed for Earth remote sensing do not always transfer cleanly to
lunar imagery. \[1\]\[2\]

------------------------------------------------------------------------

# 🧠 What Makes This Different?

Our proposed contribution is not simply another implementation of SIFT.

The system investigates three connected ideas:

### 1. Representation selection

Different illumination conditions may favor different structural
representations.

``` text
                 IMAGE PAIR
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      CLAHE       GRADIENT      EDGES
        │            │            │
        ▼            ▼            ▼
      SIFT          SIFT          SIFT
        │            │            │
        ▼            ▼            ▼
     Metrics       Metrics       Metrics
        │            │            │
        └────────────┼────────────┘
                     ▼
             BEST REPRESENTATION
```

### 2. Adaptive correspondence

Different image pairs may require different matching strategies.

``` text
                   IMAGE PAIR
                       │
                       ▼
                PAIR ANALYSIS
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      Good texture  Weak texture  Large illumination
          │            │            │
          ▼            ▼            ▼
         SIFT      Crater cues   Normalized features
          │            │            │
          └────────────┼────────────┘
                       ▼
                  CORRESPONDENCE
```

### 3. Adaptive geometry

We do not assume that one global homography is always sufficient.

``` text
Correspondences
       │
       ▼
Global model
       │
       ▼
Check residuals
   │          │
   │          └───────────────┐
   ▼                          ▼
Good enough               Spatially structured error
   │                          │
   ▼                          ▼
Use global model          Local / piecewise model
```

The governing principle is:

> **Use the simplest transformation model that adequately explains the
> observed correspondences.**

This avoids both underfitting and unnecessary geometric complexity.

------------------------------------------------------------------------

# 🛰️ End-to-End Pipeline

``` text
 ┌──────────────────────┐       ┌──────────────────────┐
 │   SOURCE / MOVING    │       │ REFERENCE / FIXED    │
 │  Chandrayaan-2 etc.  │       │ LRO / SELENE etc.    │
 └──────────┬───────────┘       └──────────┬───────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
                ┌─────────────────────┐
                │   1. PAIR ANALYSIS  │
                │ texture / scale /   │
                │ illumination /      │
                │ geometric cues      │
                └──────────┬──────────┘
                           ▼
             ┌──────────────────────────┐
             │ 2. REPRESENTATION BANK   │
             │                          │
             │ raw / CLAHE / gradient / │
             │ contrast / Laplacian /   │
             │ edges / phase / scales   │
             └────────────┬─────────────┘
                          ▼
             ┌──────────────────────────┐
             │ 3. CORRESPONDENCE        │
             │                          │
             │ SIFT / learned features /│
             │ crater cues / dense      │
             │ matching                 │
             └────────────┬─────────────┘
                          ▼
             ┌──────────────────────────┐
             │ 4. MATCH FILTERING       │
             │                          │
             │ ratio test / cross-check │
             │ / confidence filtering   │
             └────────────┬─────────────┘
                          ▼
             ┌──────────────────────────┐
             │ 5. ROBUST GEOMETRY       │
             │                          │
             │ RANSAC / MAGSAC-style    │
             │ estimation               │
             └────────────┬─────────────┘
                          ▼
             ┌──────────────────────────┐
             │ 6. MODEL SELECTION       │
             │                          │
             │ translation → similarity │
             │ → affine → homography    │
             └────────────┬─────────────┘
                          │
                global model sufficient?
                     /            \
                   YES             NO
                    │               │
                    ▼               ▼
              global warp      piecewise/local
                                    │
                                    ▼
                               H₁ H₂ ... Hₙ
                                    │
                    ┌───────────────┘
                    ▼
             ┌──────────────────────────┐
             │ 7. REFINEMENT           │
             │                          │
             │ local optimization /     │
             │ sub-pixel refinement     │
             └────────────┬─────────────┘
                          ▼
             ┌──────────────────────────┐
             │ 8. REGISTRATION OUTPUT   │
             │                          │
             │ registered image         │
             │ correspondences          │
             │ residual maps            │
             └────────────┬─────────────┘
                          ▼
             ┌──────────────────────────┐
             │ 9. CONFIDENCE / METRICS  │
             │                          │
             │ RMSE / inliers / ratio / │
             │ spatial coverage / time  │
             └──────────────────────────┘
```

------------------------------------------------------------------------

# 🔬 Stage 1 --- Image-Pair Characterization

Before selecting a registration method, the system should estimate what
makes the pair difficult.

## Texture strength

A feature-rich region may contain:

``` text
       ○      •
   •       ○
       ╲
  ○       •      ○
```

while a smooth mare region may contain little repeatable texture.

Useful signals include:

-   gradient density
-   local variance
-   edge density
-   keypoint density

## Illumination difference

We can compare:

-   intensity distributions,
-   local contrast,
-   gradient statistics,
-   shadow/bright-region statistics.

The purpose is **not** to claim exact physical illumination recovery
unless Sun geometry and photometric calibration are available.

## Scale difference

We can estimate effective scale mismatch from:

-   metadata where available,
-   image resolution / ground sampling information,
-   feature-scale statistics,
-   preliminary correspondence behavior.

## Geometric difficulty

After initial matching, we can inspect:

-   reprojection error,
-   residual distribution,
-   spatial consistency.

This helps determine whether a single global model is plausible.

------------------------------------------------------------------------

# 🌗 Stage 2 --- Terrain-Structure Representations

The key idea is to create several views of the same terrain.

## Raw grayscale

Baseline representation.

## CLAHE

Local contrast enhancement.

Useful when global brightness differences hide local terrain structure.

## Gradient magnitude

Highlights rapid spatial intensity changes.

Conceptually:

``` text
smooth terrain ─────────────
                      ↑
                 strong gradient
```

This can emphasize:

-   crater rims,
-   scarps,
-   ridges,
-   boundaries.

## Edge map

Focuses on structural boundaries rather than absolute intensity.

## Laplacian

Emphasizes second-order intensity changes and fine structure, while
requiring care because it can amplify noise.

## Local contrast

Compares a pixel or patch against its local neighborhood rather than
relying only on absolute brightness.

## Phase congruency

A more advanced structural representation that emphasizes significant
local phase relationships rather than raw intensity magnitude.

It is particularly interesting for this project because recent
planetary-registration work has explored multi-scale phase-congruency
representations under changing illumination. \[3\]

## Shadow suppression

A candidate preprocessing stage intended to reduce the effect of
illumination/shadow regions.

It should be treated as an experimental module, because aggressive
shadow removal can also destroy useful terrain boundaries.

## Multi-scale representations

Large-scale structure:

``` text
        _________
      /           \
     |   CRATER    |
      \___________/
```

Fine-scale structure:

``` text
   •  •  small craters
      •
```

Both can be useful, but they behave differently when image resolution
changes.

------------------------------------------------------------------------

# 🔎 Stage 3 --- Correspondence

Once a representation is selected, we extract corresponding terrain
features.

## Classical baseline --- SIFT

SIFT detects distinctive local structures and constructs descriptors
intended to remain useful under changes such as scale, rotation, and
moderate illumination/geometric variation. \[4\]

Conceptually:

``` text
IMAGE
  │
  ▼
keypoints
  │
  ▼
descriptors
  │
  ▼
matching
```

A keypoint answers:

> **Where is the distinctive location?**

A descriptor answers:

> **What does its local neighborhood look like?**

SIFT is our baseline, not our final research claim.

------------------------------------------------------------------------

# 🤖 Learned Correspondence

The framework is designed to support learned methods such as:

### SuperPoint

SuperPoint jointly detects interest points and computes descriptors
using a fully convolutional network. Its paper also introduces
homographic adaptation for improving repeatability. \[5\]

### LoFTR

LoFTR takes a different approach: instead of requiring a conventional
detector → descriptor → matcher sequence, it establishes coarse dense
correspondences and then refines them. The authors specifically
highlight its ability to produce matches in low-texture regions where
detector-based methods can struggle. \[6\]

This makes LoFTR an interesting candidate for lunar regions with weak
local texture.

**Important:** performance on ordinary indoor/outdoor benchmarks does
not automatically imply equivalent performance on lunar imagery. The
project therefore treats learned methods as experimental baselines to be
evaluated on the actual lunar domain.

------------------------------------------------------------------------

# 🌑 Crater-Based Correspondence

Some lunar scenes contain few strong generic keypoints but contain
recognizable crater structures.

A possible future strategy is:

``` text
Image
  ↓
Crater detection
  ↓
Center / radius / shape
  ↓
Crater geometry
  ↓
Cross-image matching
  ↓
Geometric registration
```

This is motivated by planetary navigation and registration literature in
which crater patterns are used as stable lunar landmarks. \[7\]

This module should only be considered **implemented** once an actual
crater detector and geometric matching procedure are present.

------------------------------------------------------------------------

# 🧹 Stage 4 --- Match Filtering

Raw feature matching can produce incorrect correspondences.

The baseline pipeline can use:

-   nearest-neighbor matching,
-   Lowe ratio test,
-   mutual/cross-check filtering,
-   descriptor-distance filtering.

The goal is:

``` text
Raw matches
     │
     ▼
Candidate matches
     │
     ▼
More reliable matches
```

------------------------------------------------------------------------

# 🛡️ Stage 5 --- Robust Geometry with RANSAC

Even good descriptor matching can contain false correspondences.

RANSAC estimates a geometric model while tolerating a significant
fraction of incorrect observations. It is a classic robust-estimation
method introduced for model fitting and image-analysis problems. \[8\]

Conceptually:

``` text
100 candidate matches
       │
       ▼
      RANSAC
       │
 ┌─────┴─────┐
 ▼           ▼
INLIERS    OUTLIERS
 80          20
```

The **inliers** support a common geometric explanation.

The **outliers** do not.

------------------------------------------------------------------------

# 📐 Stage 6 --- Geometric Model Selection

We do not want to blindly use a homography.

The candidate models are:

``` text
Translation
     ↓
Similarity
     ↓
Affine
     ↓
Homography
     ↓
Piecewise / local model
```

The model should become more complex only when the data requires it.

## Why?

A global homography assumes a single relationship between the images.

But lunar terrain is three-dimensional:

``` text
              /\             crater rim
             /  \
____________/    \____________
```

Different elevations can create spatially varying image displacement
when the viewpoint changes.

Therefore:

> **A single homography can fit one part of the terrain well while
> producing systematic residuals elsewhere.**

------------------------------------------------------------------------

# 🧩 Global vs Piecewise Registration

## Global model

``` text
┌─────────────────────────────┐
│                             │
│             H               │
│                             │
│    one transformation       │
│       for the image         │
│                             │
└─────────────────────────────┘
```

## Piecewise model

``` text
┌─────────┬─────────┬─────────┐
│   H₁    │   H₂    │   H₃    │
├─────────┼─────────┼─────────┤
│   H₄    │   H₅    │   H₆    │
├─────────┼─────────┼─────────┤
│   H₇    │   H₈    │   H₉    │
└─────────┴─────────┴─────────┘
```

The piecewise model is used only if the residual structure justifies it.

This prevents overfitting.

------------------------------------------------------------------------

# 📍 Stage 7 --- Spatially Distributed Matches

A transformation supported only by one small crater cluster can be
unstable.

We therefore analyze match distribution over a grid:

``` text
┌─────┬─────┬─────┬─────┐
│ ●   │     │ ●   │     │
├─────┼─────┼─────┼─────┤
│     │ ●   │     │ ●   │
├─────┼─────┼─────┼─────┤
│ ●   │     │ ●   │     │
└─────┴─────┴─────┴─────┘
```

We can measure:

-   occupied-cell fraction,
-   match count per cell,
-   spatial entropy,
-   coverage,
-   maximum local concentration.

This directly supports the SIH requirement for spatially distributed
correspondence.

------------------------------------------------------------------------

# 🎯 Stage 8 --- Registration and Refinement

Once a valid model is selected:

``` text
Source image
     │
     ▼
geometric transformation
     │
     ▼
warping
     │
     ▼
registered image
```

Then, where justified, local optimization can refine correspondence
locations beyond integer-pixel coordinates.

The system should **not claim sub-pixel accuracy merely because a
sub-pixel optimizer exists**. It must be demonstrated against
appropriate ground truth or reference measurements.

------------------------------------------------------------------------

# 📊 Stage 9 --- Evaluation

Every experiment should produce reproducible metrics.

## RMSE

For corresponding points (p_i) and predicted points (`\hat `{=tex}p_i):

\[ RMSE = `\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left\|p_i-\hat p_i\right\|^2
}`{=tex} \]

Lower is generally better.

## Inlier ratio

\[ Inlier Ratio = `\frac{N_{inliers}}`{=tex} {N\_{matches}} \]

Higher is generally better.

## Spatial coverage

Measures how widely correspondences are distributed across the image.

## Additional metrics

-   source keypoints
-   reference keypoints
-   raw matches
-   filtered matches
-   inliers
-   reprojection RMSE
-   spatial coverage
-   runtime
-   transformation-model complexity
-   registration success/failure

------------------------------------------------------------------------

# 🧪 Research Benchmark

The central benchmark should compare **representations**, not just
algorithms.

For example:

``` text
Same image pair
       │
       ├── Raw
       ├── CLAHE
       ├── Gradient
       ├── Local contrast
       ├── Laplacian
       ├── Edge
       └── Phase congruency
              │
              ▼
       Same matcher + geometry
              │
              ▼
       Compare RMSE / inliers /
       coverage / runtime
```

This lets us ask:

> **Which representation gives the most stable correspondences under
> changing lunar illumination?**

Then we can investigate whether the answer changes with terrain type,
texture level, or Sun angle.

------------------------------------------------------------------------

# 🔬 Adaptive Decision Layer

The eventual system can use interpretable rules.

### Case A --- texture-rich

``` text
Good texture
    ↓
SIFT / SuperPoint
    ↓
matching
    ↓
RANSAC
```

### Case B --- weak texture + visible craters

``` text
Weak generic texture
    ↓
Crater cues
    ↓
Geometric crater matching
```

### Case C --- strong illumination difference

``` text
Large photometric difference
    ↓
Illumination-robust representation
    ↓
Feature / learned matching
```

### Case D --- small viewpoint difference

``` text
Small geometric change
    ↓
Local feature matching
    ↓
Global model
```

### Case E --- strong spatially varying distortion

``` text
Global model
    ↓
large structured residuals
    ↓
piecewise / local registration
```

The first implementation should use **transparent rules and measurable
thresholds**, not a black-box classifier.

------------------------------------------------------------------------

# 📈 Literature Context

This project is grounded in several lines of prior research.

## Lunar / planetary image registration

Planetary image-feature research has explicitly reported that lunar
imagery can exhibit low contrast and uneven illumination, motivating
specialized feature extraction rather than blindly transferring
Earth-remote-sensing methods. \[1\]

A systematic planetary co-registration study demonstrated
multi-instrument registration across Mars and Moon datasets and focused
on robustness to varied image inputs. \[2\]

More recent work using Chandrayaan-2 lunar data has compared SIFT,
ASIFT, AKAZE, RIFT2 and SuperGlue across cross-modality lunar image
pairs, reporting that preprocessing and illumination conditions
materially affect registration performance. \[9\]

Recent lunar-image studies have also compared classical and learned
feature methods across resolution changes. \[10\]

## Illumination-robust structure

Recent planetary-registration research has investigated photometric
reliability, phase-congruency representations, brightness inversion and
shadow migration specifically for lunar multi-illumination registration.
\[3\]

This strongly supports our decision to make **terrain-structure
representation** a first-class research component rather than treating
illumination as a minor preprocessing detail.

## Learned matching

SuperPoint provides a learned detector/descriptor baseline. \[5\]

LoFTR provides a detector-free, coarse-to-fine correspondence baseline
that is particularly interesting for low-texture regions. \[6\]

------------------------------------------------------------------------

# 🖼️ Literature Figures and Research Visuals

This repository should distinguish between:

### Original project diagrams

The architecture diagrams in this README are **original diagrams created
for this project**.

### Literature figures

For scientific attribution, the project should link to --- rather than
silently copy --- figures from published work.

Recommended figures to inspect when preparing the report/pitch:

1.  **LoFTR Figure 1** --- comparison of detector-based and
    detector-free matching in low-texture regions. \[6\]
2.  **SuperPoint figures** --- learned keypoint detection and
    homographic adaptation. \[5\]
3.  **Automatic Extraction of Planetary Image Features** ---
    lunar/planetary feature representations including contour and shape
    features. \[1\]
4.  **PWIFT framework figure** --- recent planetary multi-illumination
    registration architecture using photometric reliability and phase
    congruency. \[3\]
5.  **Chandrayaan-2 comparative study** --- cross-sensor lunar
    registration experiments and preprocessing comparisons. \[9\]

When adding a copyrighted figure to a presentation or repository, check
the paper's reuse license and attribution requirements first.

------------------------------------------------------------------------

# 📚 Key Literature

### \[1\] Automatic Extraction of Planetary Image Features

Troglio, G., Le Moigne, J., Moser, S. B., Serpico, S. B., &
Benediktsson, J. A.

The work specifically discusses lunar imagery, low contrast, uneven
illumination, feature extraction and image registration.

### \[2\] A Systematic Solution to Multi-Instrument Coregistration of High-Resolution Planetary Images to an Orthorectified Baseline

Sidiropoulos, P. & Muller, J.-P.

A multi-instrument planetary co-registration framework evaluated on Mars
and Moon datasets.

### \[3\] Photometric-weighted invariant feature transform for planetary surface image registration under complex illumination

Yan, Q., Guo, Y., & Zeng, X.

A recent planetary-registration approach using photometric reliability,
phase congruency, bright/dark descriptors and homography-based cleanup.

### \[4\] Distinctive Image Features from Scale-Invariant Keypoints

Lowe, D. G., 2004.

The foundational SIFT paper.

### \[5\] SuperPoint: Self-Supervised Interest Point Detection and Description

DeTone, D., Malisiewicz, T., & Rabinovich, A., 2018.

Learned local feature detection and description.

### \[6\] LoFTR: Detector-Free Local Feature Matching with Transformers

Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X., 2021.

Detector-free coarse-to-fine feature matching.

### \[7\] Lunar Crater Identification in Digital Images

Christian, J. A., Derksen, H., & Watkins, R., 2021.

Open-access work on identifying lunar crater patterns for navigation and
related applications.

### \[8\] Random Sample Consensus

Fischler, M. A. & Bolles, R. C., 1981.

Foundational RANSAC paper.

### \[9\] Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data

Makharia, R., Singla, J. G., Amitabh, Dube, N., & Sharma, H., 2025.

Comparison of classical and learned matching on Chandrayaan-2-related
lunar data.

### \[10\] MoonMetaSync: Lunar Image Registration Analysis

Kumar, A., Kaushal, S., & Murthy, S. V., 2024.

Comparison of SIFT, ORB and a proposed feature representation across
lunar image scales.

------------------------------------------------------------------------

# 🧪 Experimental Matrix

The benchmark should systematically vary:

  Factor           Example conditions
  ---------------- ---------------------------------------------------
  Illumination     low / medium / high Sun-angle difference
  Scale            same / moderate / large scale difference
  Viewpoint        small / medium / large viewpoint change
  Texture          high / medium / low
  Terrain          crater-rich / mare / mixed
  Representation   raw / CLAHE / gradient / edge / Laplacian / phase
  Matcher          SIFT / SuperPoint / LoFTR / crater-based
  Geometry         similarity / affine / homography / piecewise
  Metrics          RMSE / inlier ratio / coverage / runtime

The first experiments should use **synthetic data with known
transformations**, followed by controlled lunar-image experiments, and
finally real Chandrayaan-2/LRO cross-sensor pairs.

------------------------------------------------------------------------

# 🧪 Synthetic Test Strategy

Before using difficult real lunar imagery, construct controlled
experiments where the ground-truth transformation is known.

``` text
Reference image
      │
      ├── brightness change
      ├── contrast change
      ├── rotation
      ├── scale
      ├── perspective
      └── local distortion
             │
             ▼
        Synthetic source
             │
             ▼
       registration system
             │
             ▼
      compare with known
       ground truth
```

This lets us distinguish:

> **algorithm failure**

from:

> **dataset / overlap / sensor / metadata problems.**

------------------------------------------------------------------------

# 📁 Project Structure

``` text
lunar-image-registration/
│
├── configs/
│   ├── default.yaml
│   ├── representations/
│   └── experiments/
│
├── data/
│   ├── synthetic/
│   ├── real/
│   │   ├── ohrc/
│   │   ├── tmc2/
│   │   ├── iirs/
│   │   ├── lro/
│   │   └── selene/
│   ├── processed/
│   ├── patches/
│   └── test/
│
├── notebooks/
│   ├── exploration/
│   └── experiments/
│
├── src/
│   └── lunar_registration/
│       ├── io/
│       ├── preprocessing/
│       │   ├── clahe.py
│       │   ├── gradient.py
│       │   ├── local_contrast.py
│       │   ├── laplacian.py
│       │   ├── edges.py
│       │   └── representations.py
│       │
│       ├── features/
│       │   ├── sift.py
│       │   ├── superpoint.py
│       │   └── loftr.py
│       │
│       ├── matching/
│       ├── geometry/
│       │   ├── models.py
│       │   ├── ransac.py
│       │   └── model_selection.py
│       │
│       ├── registration/
│       │   ├── global.py
│       │   ├── piecewise.py
│       │   └── refinement.py
│       │
│       ├── pair_analysis/
│       ├── adaptive/
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── spatial.py
│       │   └── confidence.py
│       │
│       └── utils/
│
├── scripts/
├── tests/
├── outputs/
│   ├── matches/
│   ├── registered/
│   ├── metrics/
│   └── figures/
│
├── docs/
│   ├── literature/
│   └── experiments/
│
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

------------------------------------------------------------------------

# 🚀 Installation

``` bash
git clone <repository-url>
cd lunar-image-registration

python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# ▶️ Usage

## Baseline registration

``` bash
python -m lunar_registration.cli register \
    --source data/real/source.tif \
    --reference data/real/reference.tif \
    --config configs/default.yaml
```

## Representation benchmark

``` bash
python -m lunar_registration.cli benchmark-representations \
    --source data/real/source.tif \
    --reference data/real/reference.tif
```

## Adaptive registration

``` bash
python -m lunar_registration.cli adaptive \
    --source data/real/source.tif \
    --reference data/real/reference.tif
```

## Evaluation

``` bash
python -m lunar_registration.cli evaluate \
    --config configs/default.yaml
```

> These commands describe the intended CLI. Keep them synchronized with
> the actual implementation.

------------------------------------------------------------------------

# 📦 Expected Outputs

For every image pair:

``` text
outputs/
├── registered/
│   └── registered_image.tif
│
├── matches/
│   ├── raw_matches.png
│   └── inlier_matches.png
│
├── metrics/
│   └── results.json
│
└── figures/
    ├── representation_comparison.png
    ├── residual_map.png
    └── spatial_match_distribution.png
```

Example schema:

``` json
{
  "representation": "gradient",
  "feature_method": "sift",
  "keypoints_source": 0,
  "keypoints_reference": 0,
  "raw_matches": 0,
  "inliers": 0,
  "inlier_ratio": 0.0,
  "rmse_pixels": null,
  "spatial_coverage": 0.0,
  "runtime_seconds": 0.0,
  "geometric_model": "homography",
  "confidence": null
}
```

The zero/null values are an output schema example, **not experimental
results**.

------------------------------------------------------------------------

# 🛣️ Development Roadmap

## Phase 1 --- Reliable Classical Baseline

-   [ ] dataset ingestion
-   [ ] image loading
-   [ ] grayscale conversion
-   [ ] CLAHE
-   [ ] SIFT
-   [ ] descriptor matching
-   [ ] Lowe ratio test
-   [ ] RANSAC
-   [ ] homography
-   [ ] warping
-   [ ] RMSE
-   [ ] inlier ratio
-   [ ] match visualization

## Phase 2 --- Illumination-Robust Representations

-   [ ] representation abstraction
-   [ ] CLAHE benchmark
-   [ ] gradient magnitude
-   [ ] local contrast
-   [ ] Laplacian
-   [ ] edge representation
-   [ ] multi-scale representations
-   [ ] phase congruency investigation
-   [ ] shadow suppression investigation

## Phase 3 --- Representation Benchmark

-   [ ] fixed image-pair benchmark
-   [ ] controlled illumination experiments
-   [ ] RMSE comparison
-   [ ] inlier-ratio comparison
-   [ ] spatial-coverage comparison
-   [ ] runtime comparison
-   [ ] representation ranking

## Phase 4 --- Adaptive Pair Analysis

-   [ ] texture-strength analysis
-   [ ] illumination-difference analysis
-   [ ] scale-difference analysis
-   [ ] geometric-difficulty analysis
-   [ ] interpretable strategy selection

## Phase 5 --- Alternative Correspondence

-   [ ] SuperPoint
-   [ ] learned matching
-   [ ] LoFTR
-   [ ] crater-based correspondence
-   [ ] cross-sensor experiments

## Phase 6 --- Terrain-Aware Geometry

-   [ ] translation model
-   [ ] similarity model
-   [ ] affine model
-   [ ] homography model
-   [ ] residual-map analysis
-   [ ] piecewise/grid registration
-   [ ] model complexity penalty
-   [ ] global-vs-local benchmark

## Phase 7 --- High-Precision Registration

-   [ ] local optimization
-   [ ] sub-pixel refinement
-   [ ] high-precision error evaluation
-   [ ] spatially uniform control-point selection

## Phase 8 --- Final Adaptive System

-   [ ] end-to-end adaptive pipeline
-   [ ] confidence estimation
-   [ ] automated benchmarking
-   [ ] real Chandrayaan-2 experiments
-   [ ] cross-mission experiments
-   [ ] reproducible result package
-   [ ] final documentation

------------------------------------------------------------------------

# 🧪 Scientific Validation Rules

This project should follow a strict distinction between:

### Implemented

Code exists and has been tested.

### Experimental

A method exists but its performance is still being evaluated.

### Proposed

A research direction has been designed but not implemented.

### Demonstrated

A method has been validated on a defined benchmark with reported
results.

Do **not** claim:

-   illumination invariance,
-   scale invariance,
-   sub-pixel accuracy,
-   terrain-aware superiority,
-   or improvement over existing methods

until the corresponding experiments demonstrate them.

------------------------------------------------------------------------

# 🔁 Reproducibility

Every experiment should record:

``` text
Dataset
Source sensor
Reference sensor
Image pair
Preprocessing
Representation
Feature method
Matcher
Ratio threshold
RANSAC method
Geometric model
Refinement method
Metrics
Runtime
Random seed
Software version
```

Results should be stored as machine-readable JSON/CSV files so that
experiments can be compared later.

------------------------------------------------------------------------

# 📜 Data Policy

Large scientific datasets should **not** be committed directly to Git.

Keep:

``` text
data/
```

for local data and provide:

-   dataset source,
-   acquisition metadata,
-   preprocessing instructions,
-   expected directory structure,
-   checksums where appropriate.

The project should respect the license and redistribution conditions of
every dataset, model and third-party dependency.

------------------------------------------------------------------------

# 📊 What Success Looks Like

A successful final system should not simply say:

``` text
"Registration complete."
```

It should produce something closer to:

``` text
PAIR
 ├── source: Chandrayaan-2 OHRC
 ├── reference: LRO NAC
 │
 ▼
PAIR ANALYSIS
 ├── texture: medium
 ├── illumination difference: high
 └── spatial distortion: moderate
 │
 ▼
SELECTED REPRESENTATION
 └── gradient + local contrast
 │
 ▼
CORRESPONDENCE
 ├── candidates: ...
 └── reliable matches: ...
 │
 ▼
GEOMETRY
 ├── global homography: insufficient
 └── piecewise model: selected
 │
 ▼
RESULT
 ├── RMSE: measured experimentally
 ├── inlier ratio: measured experimentally
 ├── spatial coverage: measured experimentally
 └── confidence: measured experimentally
```

The values must come from the actual experiment.

------------------------------------------------------------------------

# 🔭 Research Direction

The long-term research hypothesis is:

> **Lunar image registration can be made more robust by adapting the
> representation, correspondence method, and geometric model to the
> observable characteristics of the image pair rather than applying one
> fixed registration pipeline to every pair.**

This leads to three primary research questions:

### RQ1 --- Representation

**Which terrain-structure representation produces the most stable
correspondences under lunar illumination changes?**

### RQ2 --- Adaptation

**Can observable pair characteristics predict which correspondence
strategy is most reliable?**

### RQ3 --- Geometry

**When does a global geometric model fail on non-planar lunar terrain,
and when does a piecewise model provide a meaningful improvement without
overfitting?**

------------------------------------------------------------------------

# 🏆 Intended Contribution

The intended contribution is therefore **not**:

> "We implemented SIFT."

Instead:

> **We develop and evaluate an adaptive lunar image-registration
> framework that treats illumination-robust representation selection,
> correspondence strategy selection, and global-versus-local geometric
> model selection as explicit parts of the registration problem.**

SIFT, SuperPoint, LoFTR, RANSAC, homography and crater matching are
components/baselines within that framework.

------------------------------------------------------------------------

# 📜 License

This project is released under the **MIT License**.

The MIT License applies to this project's original source code and
documentation.

It does **not** automatically grant rights to redistribute:

-   Chandrayaan-2 data,
-   LRO data,
-   SELENE/Kaguya data,
-   pretrained model weights,
-   third-party libraries,
-   or figures from external publications.

Those materials remain subject to their respective licenses and usage
conditions.

See [`LICENSE`](LICENSE) for the full license text.

------------------------------------------------------------------------

# 🙏 Acknowledgements

Developed as part of **Smart India Hackathon 2026** for the Indian Space
Research Organisation (ISRO), Department of Space.

**Problem Statement:** 26166

The project builds on established work in computer vision, planetary
image registration, lunar feature extraction, robust geometric
estimation, and learned feature matching.

------------------------------------------------------------------------

# 📚 References

1.  G. Troglio, J. Le Moigne, G. Moser, S. B. Serpico, and J. A.
    Benediktsson, **"Automatic Extraction of Planetary Image
    Features."**
2.  P. Sidiropoulos and J.-P. Muller, **"A Systematic Solution to
    Multi-Instrument Coregistration of High-Resolution Planetary Images
    to an Orthorectified Baseline,"** IEEE TGRS, 2017. DOI:
    `10.1109/TGRS.2017.2734693`.
3.  Q. Yan, Y. Guo, and X. Zeng, **"Photometric-weighted invariant
    feature transform for planetary surface image registration under
    complex illumination,"** Aerospace Science and Technology, 2026.
    DOI: `10.1016/j.ast.2026.113462`.
4.  D. G. Lowe, **"Distinctive Image Features from Scale-Invariant
    Keypoints,"** International Journal of Computer Vision, 2004. DOI:
    `10.1023/B:VISI.0000029664.99615.94`.
5.  D. DeTone, T. Malisiewicz, and A. Rabinovich, **"SuperPoint:
    Self-Supervised Interest Point Detection and Description,"** CVPR
    Workshops, 2018.
6.  J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, **"LoFTR:
    Detector-Free Local Feature Matching with Transformers,"**
    CVPR, 2021. DOI: `10.1109/CVPR46437.2021.00881`.
7.  J. A. Christian, H. Derksen, and R. Watkins, **"Lunar Crater
    Identification in Digital Images,"** Journal of the Astronautical
    Sciences, 2021.
8.  M. A. Fischler and R. C. Bolles, **"Random Sample Consensus: A
    Paradigm for Model Fitting with Applications to Image Analysis and
    Automated Cartography,"** Communications of the ACM, 1981. DOI:
    `10.1145/358669.358692`.
9.  R. Makharia, J. G. Singla, Amitabh, N. Dube, and H. Sharma,
    **"Comparative Evaluation of Traditional and Deep Learning Feature
    Matching Algorithms using Chandrayaan-2 Lunar Data,"** 2025.
10. A. Kumar, S. Kaushal, and S. V. Murthy, **"MoonMetaSync: Lunar Image
    Registration Analysis,"** 2024.

------------------------------------------------------------------------

## Project Status

🚧 **In Development**

The baseline registration system is being developed first. Advanced
adaptive representation selection, terrain-aware local registration,
crater-based matching, learned matching, and sub-pixel refinement will
be introduced and validated incrementally.

> **Build the baseline. Measure it. Break it. Improve it. Then prove the
> improvement.**
