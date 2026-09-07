# Experiment Tracking & Benchmark Execution Ledger

**Project:** Multi-modal, Sun-angle and Scale-invariant Image Correspondence using Chandrayaan-2 Optical Images  
**Role Owner:** Member B (Evaluation) & Member E (Baseline Execution)  
**Governance:** Governed by [docs/research/EVALUATION_PROTOCOL.md](file:///Users/prachichoubey/Desktop/Projects/lunar-image-registration/docs/research/EVALUATION_PROTOCOL.md)  

---

## 1. Experiment Registry Overview

All quantitative evaluations are recorded in this ledger with their unique experiment ID, configuration file, target dataset pair, hardware environment, and metric outcomes.

| Experiment ID | Phase | Configuration File | Target Pair / Test | Method Description | Inlier Ratio ($R_{inlier}$) | Reproj. RMSE (px) | Spatial Coverage ($C_{spatial}$) | Success? | Status / Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`EXP-SANITY-01`** | Phase 0 | Synthetic Audit Script | Synthetic Crater Patch | SIFT + BF + RANSAC (Audit) | **98.2%** | **0.212** | 0.812 | Pass | Completed in `docs/engineering/BASELINE_AUDIT.md` |
| **`EXP-DEV01-SIFT`** | Phase 2 | `configs/sift.yaml` | `DEV-01` (TMC-2 to LRO NAC) | Classical SIFT + BF + MAGSAC++ | *TBD* | *TBD* | *TBD* | *Pending* | Awaiting Member D Pilot Manifest |
| **`EXP-DEV02-SCALE`**| Phase 2 | `configs/sift.yaml` | `DEV-02` (OHRC to LRO NAC) | SIFT + Metadata Scale Downsample | *TBD* | *TBD* | *TBD* | *Pending* | Awaiting Member D Pilot Manifest |
| **`EXP-DEV03-STEREO`**| Phase 2 | `configs/sift.yaml` | `DEV-03` (TMC-2 Fore / Aft) | SIFT + Affine Warping | *TBD* | *TBD* | *TBD* | *Pending* | Awaiting Member D Pilot Manifest |
| **`EXP-DEV04-IR`** | Phase 3 | `configs/default.yaml`| `DEV-04` (TMC-2 to IIRS) | Phase Congruency (RIFT) | *TBD* | *TBD* | *TBD* | *Pending* | Phase 3 Invariance Gate |
| **`EXP-LG-BENCH`** | Phase 4 | `configs/superpoint.yaml`| `TEST-01` to `TEST-08` | SuperPoint + LightGlue + MAGSAC++ | *TBD* | *TBD* | *TBD* | *Pending* | Phase 4 Learned Gate |
| **`EXP-LOFTR-MARE`** | Phase 4 | `configs/loftr.yaml` | Low-contrast Mare Subset | Dense LoFTR + Grid Tiling | *TBD* | *TBD* | *TBD* | *Pending* | Phase 4 Learned Gate |

---

## 2. Standardized Execution Command
Every experiment must be invoked via the CLI using a versioned configuration file to ensure exact reproducibility:

```bash
# Example benchmark run command:
python -m lunar_registration.cli register \
    --source data/external/ch2/pilot/tmc2_nadir_patch.tif \
    --reference data/external/lro/pilot/lroc_nac_patch.tif \
    --config configs/sift.yaml \
    --output-dir outputs/ \
    --experiment-id EXP-DEV01-SIFT
```

---

## 3. Results Artifact Locations
- Numerical Metrics (JSON): `outputs/metrics/{experiment_id}.json`
- Match Visualizations (PNG): `outputs/matches/{experiment_id}_matches.png`
- Registered GeoTIFFs: `outputs/registered/{experiment_id}_registered.tif`
- High-Resolution Figure Plots: `outputs/figures/{experiment_id}_evaluation.png`
