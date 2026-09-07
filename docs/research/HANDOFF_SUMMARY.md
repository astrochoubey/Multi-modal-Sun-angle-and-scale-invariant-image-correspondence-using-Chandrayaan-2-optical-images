# Research Handoff Summary: Operational Directives for Engineering & Pitch

**Task ID:** B-01 (Step 5 Handoff Deliverable)  
**Author:** Member B — Research, Novelty, and Evaluation Lead  
**Phase:** Phase 0 — Research and Requirements  
**Date:** 2026-09-07  
**Distribution:** Members A, C, D, E, and F  

---

## 1. Handoff to Member D (Data, Metadata, and Planetary Lead)

### Required Data Conditions for 4-Pair Pilot Manifest
Member D must ensure the initial pilot dataset satisfies the four specific test regimes needed for scientific baseline validation:

| Pair ID | Source Sensor & Mode | Reference Sensor | Target Overlap | Critical Metadata Parameters Required | Purpose & Scientific Evaluation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`DEV-01`** | Chandrayaan-2 **TMC-2 Nadir** (5.0 m GSD) | NASA **LRO LROC NAC** (downsampled to 5.0 m) | $\ge 60\%$ | Solar incidence $\Delta i \le 15^\circ$, Azimuth $\Delta \phi \le 20^\circ$, PDS4 XML labels | Sanity verification; baseline classical SIFT / ORB benchmark under benign illumination. |
| **`DEV-02`** | Chandrayaan-2 **OHRC** (0.25 m GSD) | NASA **LRO LROC NAC** (0.5–1.0 m GSD) | $\ge 50\%$ | GSD ratio $2\times$ to $4\times$, Sub-solar coordinates, Emission angle $\le 10^\circ$ | Scale-disparity pilot; testing multi-octave pyramid matching and keypoint repeatability. |
| **`DEV-03`** | Chandrayaan-2 **TMC-2 Fore** ($+25^\circ$) | Chandrayaan-2 **TMC-2 Aft** ($-25^\circ$) | $\ge 80\%$ | Along-track stereo time offset, incidence/azimuth identical | In-sensor stereo baseline; tests affine and perspective distortion without shadow inversion. |
| **`DEV-04`** | Chandrayaan-2 **TMC-2 Nadir** (5.0 m GSD) | Chandrayaan-2 **IIRS** (~80 m GSD, Band @ 1.5 $\mu\text{m}$) | $\ge 70\%$ | Core footprint coordinates, band wavelength center, radiometric calibration units | Multimodal cross-spectral pilot; tests structural phase matching (RIFT) against SIFT failure. |

> [!IMPORTANT]
> **Data Integrity Checklist for Member D:**
> 1. Store raw uncompressed images strictly under `data/external/` (never commit raw data to Git).
> 2. Ensure every image file has an associated `.xml` or metadata header detailing: `SOLAR_INCIDENCE_ANGLE`, `SOLAR_AZIMUTH_ANGLE`, `GROUND_SAMPLING_DISTANCE`, and `FOOTPRINT_BOUNDING_BOX`.
> 3. Verify files are non-zero byte and log SHA-256 checksums in `data/manifests/pilot_manifest.csv`.

---

## 2. Handoff to Member E (Classical Registration Engineer)

### Immediate Engineering Guidelines for Phase 2 Implementation
1. **Dynamic Grayscale & Bit-Depth Ingestion:**
   - Modify `src/lunar_registration/preprocessing/preprocessing.py` to inspect input shape and dtype.
   - If image is 2D (single-channel) or 16-bit unsigned int (`uint16`), bypass `cv2.COLOR_BGR2GRAY` and apply percentile-based robust min-max normalization:
     $$\tilde{I} = \text{clip}\left(\frac{I - P_{1}(I)}{P_{99}(I) - P_{1}(I)}, 0, 1\right) \times 255 \quad \rightarrow \text{uint8}$$
   - This resolves the fatal OpenCV channel crash identified in `docs/engineering/BASELINE_AUDIT.md`.
2. **Decouple and Parameterize the Baseline:**
   - Ingest parameters from `configs/default.yaml` and `configs/sift.yaml`.
   - Parameterize: SIFT `nfeatures` (default 5000), Lowe's ratio threshold (default 0.75), RANSAC reprojection threshold (default 3.0 px), and estimator type (integrate `cv2.USAC_MAGSAC`).
3. **Formal Metric Reporting:**
   - Implement the exact JSON logging schema defined in Section 5 of `docs/research/EVALUATION_PROTOCOL.md`.

---

## 3. Handoff to Member F (Geometry, Learned Matching, & Precision Engineer)

### Advanced Method Benchmark Constraints & Fair-Comparison Rules
1. **Primary Candidate Selection:**
   - **Attentive Sparse Matching:** LightGlue (Lindenberger et al., 2023) paired with SuperPoint features. Permissive Apache 2.0 license, memory efficient, fast inference.
   - **Textureless Dense Matching:** LoFTR (Sun et al., 2021) for low-contrast lunar mare plains. Must be implemented with a patch-tiling wrapper to avoid out-of-memory crashes on large swaths.
   - **Structural Phase Representation:** Phase Congruency / log-Gabor filter response for shadow-inverted pairs.
2. **Spatial Coverage Grid Enforcement:**
   - Divide the reference image into an $8 \times 8$ uniform grid.
   - Implement inlier filtering that enforces maximum representation caps per cell and reports the spatial coverage ratio:
     $$C_{spatial} = \frac{\text{Occupied Cells}}{\text{Eligible Cells}} \ge 0.30$$
3. **Sub-Pixel Refinement Engine:**
   - Implement localized matrix-multiply DFT (Guizar-Sicairos et al., 2008) on $32 \times 32$ image patches around established inliers.
   - Restrict refinement strictly to inliers that pass geometric consensus; do not claim sub-pixel accuracy on unverified raw keypoints.

---

## 4. Handoff to Member C (Presentation, Visual Design, & Demo Lead)

### Core Pitch Narrative & Slide Architecture
1. **The Core Problem Hook:** The Moon has no atmosphere. Shadows are razor-sharp. When the Sun angle shifts between orbits, crater shadows invert, causing standard computer vision (SIFT, ORB) to suffer near-total matching failure ($< 3\%$ inlier ratio).
2. **The Chandrayaan-2 Scale Challenge:** Chandrayaan-2 OHRC has a resolution of 0.25 m/pixel. International reference basemaps and regional cameras (TMC-2) have resolutions of 5.0 m/pixel—a $20\times$ scale divergence.
3. **Our System Differentiator:** 
   *"A metadata-guided structural-and-learned correspondence pipeline with explicit spatial coverage and uncertainty reporting on a curated Chandrayaan-2-to-lunar-reference benchmark."*
4. **Mandatory Visual Elements to Request from Engineering:**
   - Side-by-side match visualization comparing SIFT failure vs. Phase Congruency / LightGlue success on a shadow-inverted crater.
   - An $8 \times 8$ Spatial Coverage Heatmap showing uniform match distribution across the entire scene rather than clustering on one rim.
   - A registered alpha-blended or checkerboard overlay proving visual alignment of crater walls and boulder fields.

---

## 5. Review Checklist for Member A (Team Lead Gate Approval)

Member A may officially sign off on the completion of **Task B-01** based on the following verified deliverables:
- [x] **14 Annotated Citations** documented in `docs/research/BIBLIOGRAPHY.md` conforming to the 11-field schema.
- [x] **8-Method Comparison Matrix** in `docs/research/PRIOR_WORK_MATRIX.md` evaluating illumination, scale, geometry, licenses, and gaps.
- [x] **Mathematical Evaluation Protocol** in `docs/research/EVALUATION_PROTOCOL.md` specifying formulas for $RMSE$, $R_{inlier}$, $C_{spatial}$, repeatability curves, and locked data splits.
- [x] **Comprehensive Research Dossier** in `docs/research/RESEARCH_DOSSIER.md` providing in-depth physical and photogrammetric grounding.
- [x] **One-Page Operational Handoff** in `docs/research/HANDOFF_SUMMARY.md` assigning concrete directives to Members C, D, E, and F.
