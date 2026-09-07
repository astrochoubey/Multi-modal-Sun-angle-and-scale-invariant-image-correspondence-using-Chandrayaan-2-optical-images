# Prior-Work Comparison Matrix

**Task ID:** B-01 (Step 3)  
**Author:** Member B — Research, Novelty, and Evaluation Lead  
**Phase:** Phase 0 — Research and Requirements  
**Date:** 2026-09-07  
**Status:** Complete  

---

## 1. Algorithmic Comparison Matrix

The table below benchmarks eight representative feature detection, matching, and geometric registration paradigms across the exact nine criteria defined in the *Team Task Playbook*.

| Method | Sensor / Modalities | Illumination Robustness | Scale Handling | Geometry Model | Spatial Coverage | Sub-Pixel Support | Benchmark Dataset | License & Code | Critical Gap for Our Project |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SIFT + BF + Ratio Test** *(Lowe, 2004)* | Monomodal optical; fails across spectral bands | **Poor:** Invariant only to uniform affine illumination $I' = \alpha I + \beta$; fails under shadow inversion ($> 30^\circ$ solar azimuth delta) | **Moderate:** Scale-space Gaussian octave pyramid handles up to $\approx 2.5\times$ to $3\times$ scale divergence | Planar homography ($3 \times 3$) or Affine via OpenCV RANSAC | **Clustered:** Prone to clustering in high-contrast crater rims; leaves flat lunar mare unrepresented | Yes (DoG Taylor series quadratic interpolation, $\approx 0.1$ px) | Standard terrestrial benchmarks (Mikolajczyk, Oxford) | BSD / Open Source in `cv2` | Catastrophic failure when crater shadows reverse direction; unable to bridge OHRC (0.25 m) to TMC-2 (5 m) 20x scale jump. |
| **ASIFT + RANSAC** *(Morel & Yu, 2009)* | Monomodal optical | **Poor:** Inherits raw gradient limitations of SIFT; no shadow inversion invariance | **Moderate:** Handles scale changes up to $\approx 3\times$; simulates multi-tilt parameters | Affine / Planar homography ($H$) | **Clustered:** Concentrates features on high-gradient topography edges | Yes (inherited from SIFT DoG interpolation) | Oblique aerial datasets, architectural scenes | LGPL-3.0 / C++ & Python bindings | Extreme computational latency ($\mathcal{O}(T \times N_{oct})$); does not address lighting shifts or multimodal radiometric distortion. |
| **RIFT (Radiation-variation Insensitive)** *(Li et al., 2020)* | Cross-modal (Optical-to-Infrared, Optical-to-SAR, LiDAR) | **Excellent:** Uses log-Gabor Phase Congruency (PC) and Maximum Index Map (MIM); immune to monotonic and non-monotonic radiation shifts | **Low:** Weak intrinsic scale invariance; relies on fixed-scale log-Gabor wavelets | Planar homography ($H$) via RANSAC or FSC | **Moderate:** Structural edges are well captured across image boundaries | No intrinsic sub-pixel support (relies on FAST/Harris integer coordinates) | Multi-sensor remote sensing benchmarks (RGB, thermal, SAR) | Academic / Non-commercial research license (MATLAB / Python port) | High runtime overhead from FFT filter banks; scale invariance is severely limited without external image pyramiding. |
| **OS-SIFT** *(Xiang et al., 2018)* | Optical-to-SAR, cross-sensor remote sensing | **Moderate to High:** Employs ratio-based and multi-scale gradient operators | **Moderate:** Scale-space pyramid handles $\approx 2\times$ scale disparity | Affine / Homography ($H$) via RANSAC | **Moderate:** Matches distributed across dominant structural boundaries | Yes (Taylor series expansion around extrema) | TerraSAR-X, Gaofen-2, aerial optical/SAR | Open-source research implementation | Geared specifically toward SAR speckle noise models; does not account for pushbroom scan-line distortion or high solar incidence crater morphology. |
| **SuperPoint + SuperGlue** *(DeTone 2018; Sarlin 2020)* | Monomodal and cross-day multi-view optical | **High:** Attentional graph neural network reasons globally about spatial context | **Moderate:** Handles up to $\approx 3\times$ scale change; struggles on extreme scale jumps ($> 4\times$) | Epipolar / Essential matrix ($E$) or Homography ($H$) | **Good:** SuperPoint interest points are well distributed via non-maximal suppression (NMS) | Yes (SuperPoint soft argmax keypoint head) | MegaDepth (outdoor), ScanNet (indoor) | **Non-commercial license** (Magic Leap proprietary on SuperGlue) | Proprietary license limits open-source deployment; deep network trained exclusively on terrestrial architecture; high GPU memory consumption. |
| **SuperPoint + LightGlue** *(Lindenberger et al., 2023)* | Multi-view optical, adaptable front-ends | **High:** Adaptive self- and cross-attention heads infer correspondence through feature context | **Moderate:** Up to $\approx 3\times$ to $4\times$ scale difference; requires multi-scale image pyramids for $20\times$ | Planar homography ($H$) or Epipolar via USAC / MAGSAC++ | **High:** Adaptive early-exit and thresholding retain high-confidence features across whole scene | Yes (continuous coordinate predictions from front-end) | MegaDepth, ScanNet, Image Matching Benchmark | **Apache 2.0 (Permissive)** | Front-end detector domain gap on lunar craters; requires GPU acceleration for batch swath processing. |
| **LoFTR (Detector-Free Transformer)** *(Sun et al., 2021)* | Optical, day-to-night, low-texture surfaces | **Very High:** Global self/cross-attention bypasses keypoint detector failure on textureless regions | **Low to Moderate:** Coarse feature matching at $1/8$ resolution degrades if scale difference exceeds $3\times$ | Essential matrix ($E$) or Homography ($H$) | **Excellent:** Generates uniform dense correspondence grids even in featureless lunar mare | Yes (Fine-level correlation expectation, $\approx 0.05$ px) | MegaDepth, ScanNet | Apache 2.0 (Open Source) | Massive VRAM requirements ($\ge 8$ GB VRAM for $> 800\times 800$); cannot ingest full $10,000 \times 4,000$ Chandrayaan swaths without hierarchical tiling. |
| **Matrix-Multiply DFT Sub-pixel Correlation** *(Guizar-Sicairos, 2008)* | Optical and multispectral local image patches | **Moderate:** Resilient to uniform intensity attenuation; degrades under shadow inversion | **None:** Strictly assumes zero scale difference between local patches | Pure translation ($[t_x, t_y]$) | **Local:** Operates within pre-extracted candidate patch pairs | **Outstanding:** Sub-pixel accuracy down to $1/50$th to $1/100$th of a pixel | Synthetic & microscopic registration sets | Permissive Open Source | Purely translational local operator; cannot solve global rotation, perspective distortion, or severe topography relief independently. |

---

## 2. In-Depth Trade-Off & Failure Mode Analysis

### 2.1 The Illumination Breakdown Phenomenon on Lunar Regolith
- **The Physical Mechanism:** Lunar regolith exhibits non-Lambertian, backscatter-dominated Hapke reflectance. A crater illuminated at low Sun elevation ($15^\circ$ elevation / $75^\circ$ incidence) casts long internal and external shadows. When re-observed at high Sun elevation ($60^\circ$ elevation), the interior shadow completely vanishes, and the rim appears as a diffuse circular albedo boundary.
- **Why Gradient Baselines Fail:** As documented in `docs/engineering/BASELINE_AUDIT.md`, SIFT and ORB rely on the sign and orientation of pixel gradients ($\nabla I = [\partial I / \partial x, \partial I / \partial y]^T$). When the Sun azimuth flips by $180^\circ$, gradient vectors rotate by $180^\circ$, driving the descriptor Euclidean distance outside Lowe's ratio threshold ($d_1 / d_2 > 0.75$).
- **The Phase Solution:** Structural representations such as Kovesi's Phase Congruency and RIFT's Maximum Index Map (MIM) depend on local Fourier phase alignment rather than gradient orientation, rendering crater boundary responses symmetric and invariant to lighting direction.

### 2.2 The Scale Disparity Dilemma (OHRC vs. TMC-2)
- **The Operational Reality:** 
  - Chandrayaan-2 **OHRC** has a nadir Ground Sampling Distance (GSD) of **0.25 meters/pixel**.
  - Chandrayaan-2 **TMC-2** has a nadir GSD of **5.0 meters/pixel**.
  - This corresponds to a **$20\times$ scale divergence** ($1$ pixel in TMC-2 covers $400$ pixels in OHRC).
- **Algorithmic Limits:** Standard scale-space pyramids in SIFT, SuperPoint, and LoFTR accommodate scale factors up to $2\times$ to $3\times$. No single-pass matching algorithm in the literature successfully matches a $20\times$ downsampled patch directly without:
  1. Metadata-guided downsampling / footprint projection using orbital metadata (GSD, footprint bounding box).
  2. Multi-tier hierarchical pyramid matching (matching TMC-2 to downsampled OHRC, then projecting correspondences back to native OHRC resolution).

### 2.3 The Spatial Clustering Problem
- **The Observation:** Both classical detectors (SIFT, Harris, FAST) and learned detectors (SuperPoint) concentrate $80\%+$ of their keypoints in high-contrast rugged zones (e.g. sharp crater rims, boulder clusters), completely starving flat mare basalt plains of tie points.
- **The Consequence:** When RANSAC fits a homography or affine transformation, the estimated transformation matrix is heavily biased toward the cluster. The opposite corner of the image suffers reprojection errors of 10–50 pixels because it lacks constraining inliers.
- **The Required Mitigation:** Mandatory **Spatial Coverage Grid Enforcement** (e.g. dividing image into an $8 \times 8$ grid, enforcing a cap and minimum representation per cell via Adaptive Non-Maximal Suppression or bucketing).

---

## 3. License & Hardware Constraint Evaluation

| Method Family | Key Library | Open License? | GPU Required? | Real-Time Capable ($< 1$s)? | Open-Source Recommendation for Project |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical Baseline** | OpenCV `cv2` | Yes (Apache 2.0 / BSD) | No (CPU only) | Yes (~250 ms) | **Keep as core baseline** for universal fallback and baseline benchmarking. |
| **Phase / Structural** | RIFT / Phase Congruency | Academic only (custom) | Optional | No (~3–8 s per tile) | Re-implement lightweight log-Gabor phase representation in NumPy/PyTorch. |
| **Learned Attentive** | LightGlue (Lindenberger 2023) | **Yes (Apache 2.0)** | Recommended | Yes (~50 ms on GPU) | **Primary advanced matching candidate** for Phase 4 integration. |
| **Learned Dense** | LoFTR (Sun 2021) | **Yes (Apache 2.0)** | **Mandatory** ($\ge 8$ GB) | No (~1.5–3 s per tile) | Candidate for low-contrast mare patches; restrict to localized tiles. |
| **Robust Geometry** | MAGSAC++ (`cv2.USAC_MAGSAC`) | Yes (Apache 2.0) | No (CPU only) | Yes (~10 ms) | **Immediate drop-in replacement** for standard OpenCV RANSAC. |

---

## 4. Synthesis & Architectural Conclusion

No single off-the-shelf algorithm completely solves the Chandrayaan-2 multi-modal, sun-angle, and scale-invariant registration task:
1. **Classical SIFT** provides the transparent, lightweight foundation, but fails under illumination shifts and large scale gaps.
2. **Phase Congruency / RIFT** provides the missing mathematical invariance to shadow inversion and spectral shifts, but lacks high-order multi-scale pyramid capabilities.
3. **LightGlue / LoFTR** provides state-of-the-art context-aware correspondence, but requires careful image tiling and metadata-guided scale normalization to handle satellite swaths.
4. **MAGSAC++** provides the mathematically optimal consensus estimator to replace heuristic RANSAC.

Therefore, our project must implement a **staged, metadata-guided hybrid pipeline**:
- **Stage 1:** Orbital metadata footprint normalization (aligning scales and coarse bounding boxes).
- **Stage 2:** Illumination-robust preprocessing (CLAHE + structural edge preservation).
- **Stage 3:** Dual-engine feature correspondence (SIFT baseline vs. LightGlue learned vs. Phase Congruency structural).
- **Stage 4:** Spatial-grid-constrained USAC-MAGSAC++ geometry estimation.
- **Stage 5:** Localized matrix-multiply DFT sub-pixel tie-point refinement.
