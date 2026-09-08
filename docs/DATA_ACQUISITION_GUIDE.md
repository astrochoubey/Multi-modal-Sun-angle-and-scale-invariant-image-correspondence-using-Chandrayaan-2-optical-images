# Planetary Data Acquisition & Archival Management Guide

**Task ID:** D-01 (Part 1 & Governance)  
**Author:** Member D — Data, Metadata, and Planetary-Geospatial Lead  
**Phase:** Phase 1 — Data Foundation  
**Date:** 2026-09-07  
**Reviewers:** Member A (Team Lead) & Member B (Evaluation Lead)  

---

## 1. Scope & Core Objectives

This guide establishes the protocols, data standards, directory architectures, and checksum verification routines for acquiring and curating planetary orbital imagery from the **Indian Space Research Organisation (ISRO)** and the **National Aeronautics and Space Administration (NASA)**.

Engineering and evaluation depend strictly upon original, uncorrupted, and accurately geo-referenced planetary data. Synthetic or unverified web images must never be substituted for flight data.

---

## 2. Supported Planetary Instruments & Archives

### 2.1 Chandrayaan-2 Instruments (ISRO ISSDC)
- **Archive Portal:** ISSDC PRADAN (Planetary Data Archive): [https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/)
- **Data Standard:** Planetary Data System Version 4 (**PDS4**).
- **Target Instruments:**
  1. **OHRC (Orbiter High Resolution Camera):**
     - Nadir Ground Sampling Distance (GSD): **0.25 m/pixel** from 100 km orbit.
     - Calibrated Product Level: Level-2 (radiometrically calibrated, top-of-atmosphere radiance).
     - Component Files: `.xml` (PDS4 product label), `.img` (uncompressed raw binary raster array), `.csv` (ancillary spacecraft geometry and solar vectors).
  2. **TMC-2 (Terrain Mapping Camera-2):**
     - Nadir GSD: **5.0 m/pixel**; Along-track stereo: Fore ($+25^\circ$), Nadir ($0^\circ$), Aft ($-25^\circ$).
     - Calibrated Product Level: Level-2 Calibrated Reflectance.
     - Component Files: `.xml`, `.img`, ancillary geometry tables.
  3. **IIRS (Imaging Infrared Spectrometer):**
     - GSD: $\approx 80\text{ m/pixel}$, 256 spectral bands ($0.8–5.0\,\mu\text{m}$).
     - *Note:* Excluded from initial Phase 1 pilot per Task D-01 rules; reserved for Phase 3 cross-spectral benchmarks.

### 2.2 Lunar Reconnaissance Orbiter (NASA PDS / LROC)
- **Archive Portal:** NASA LROC QuickMap ([https://quickmap.lroc.asu.edu/](https://quickmap.lroc.asu.edu/)) & PDS Planetary Image Atlas ([https://pds-imaging.jpl.nasa.gov/](https://pds-imaging.jpl.nasa.gov/)).
- **Data Standard:** PDS3 / PDS4.
- **Target Instrument:**
  1. **LROC NAC (Narrow Angle Camera - NAC-L and NAC-R):**
     - GSD: **0.5 m to 1.2 m/pixel** depending on orbital altitude.
     - Calibrated Product Level: Calibrated Radiance (`.IMG` / `.xml`).

---

## 3. Storage Hierarchy & Version Control Rules

Raw planetary images range from 500 MB to 5 GB per swath. In strict adherence to repository guidelines:
1. **Zero Git Staging of Raw Files:** `data/external/` is listed in `.gitignore`. No raw planetary archive, `.img`, `.IMG`, `.tar`, or `.zip` file may ever be committed to Git.
2. **Local Directory Hierarchy:**
   ```
   data/
   ├── external/                      # NEVER COMMITTED TO GIT
   │   ├── ch2/
   │   │   ├── pilot/                 # Task D-01 4-pair pilot sources (OHRC & TMC-2)
   │   │   └── challenge/             # Locked test set scenes
   │   └── lro/
   │       ├── pilot/                 # Task D-01 4-pair pilot references (LROC NAC)
   │       └── challenge/             # Reference swaths for locked test
   ├── manifests/                     # COMMITTED TO GIT
   │   ├── pilot_manifest.csv         # Master metadata ledger for DEV-01 to DEV-04
   │   └── test_manifest.csv          # Master metadata ledger for TEST-01 to TEST-08
   ├── processed/                     # Extracted, normalized GeoTIFF / NumPy patches
   └── test/
       └── ground_truth/              # Verified tie-point CSVs
   ```
3. **Storage Capacity Verification:**
   - Minimum free disk space requirement: **100 GB** (Verified on host system: 130 GB free on `/System/Volumes/Data`).

---

## 4. Master Manifest Schema Specification

Every acquired pair must be cataloged in `data/manifests/pilot_manifest.csv` using the following standardized columns:

| Column Name | Data Type | Example Value | Description |
| :--- | :--- | :--- | :--- |
| `pair_id` | String | `DEV-01` | Unique development or test pair identifier |
| `source_product_id` | String | `ch2_tmc_ncn_20200115T083012123_d_img_d18` | ISRO PDS4 Source Product ID |
| `source_payload` | String | `TMC2_NADIR` | Instrument and channel mode (`OHRC`, `TMC2_NADIR`, `TMC2_FORE`, `TMC2_AFT`) |
| `source_gsd_m` | Float | `5.00` | Ground Sampling Distance at nadir in meters |
| `source_solar_incidence_deg` | Float | `62.4` | Solar incidence angle at scene center ($^\circ$) |
| `source_solar_azimuth_deg` | Float | `112.5` | Solar azimuth angle measured clockwise from North ($^\circ$) |
| `source_emission_deg` | Float | `0.8` | Spacecraft emission/look angle ($^\circ$) |
| `ref_product_id` | String | `M1142493921RC` | NASA LROC NAC Product ID or reference source |
| `ref_payload` | String | `LROC_NAC` | Reference sensor name |
| `ref_gsd_m` | Float | `1.15` | Reference GSD in meters |
| `ref_solar_incidence_deg` | Float | `68.1` | Reference solar incidence angle ($^\circ$) |
| `ref_solar_azimuth_deg` | Float | `124.0` | Reference solar azimuth angle ($^\circ$) |
| `sun_angle_delta_deg` | Float | `12.8` | Absolute difference in illumination direction ($^\circ$) |
| `scale_ratio` | Float | `4.35` | Ratio of reference GSD to source GSD ($GSD_{ref} / GSD_{src}$) |
| `center_latitude` | Float | `-70.85` | Planetocentric latitude of scene center ($^\circ$) |
| `center_longitude` | Float | `42.10` | Planetocentric East longitude ($^\circ$) |
| `source_local_path` | String | `data/external/ch2/pilot/dev01_source.img` | Relative path to local source image |
| `source_xml_path` | String | `data/external/ch2/pilot/dev01_source.xml` | Relative path to local source PDS4 label |
| `ref_local_path` | String | `data/external/lro/pilot/dev01_ref.IMG` | Relative path to local reference image |
| `sha256_source` | String | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | SHA-256 checksum of source image file |
| `sha256_ref` | String | `f4c8996fb92427ae41e4649b934ca495991b7852b855e3b0c44298fc1c149afb` | SHA-256 checksum of reference image file |

---

## 5. Checksum & Data Integrity Verification Protocol

To prevent corrupted downloads from invalidating engineering benchmarks:
1. Immediately upon completing a download, Member D executes:
   ```bash
   shasum -a 256 <downloaded_file>
   ```
2. The generated hash must be cross-referenced against the checksum reported in the PDS4 XML label (`<File_Area_Observational><File><md5_checksum>` or `<sha256_checksum>`).
3. If an archive fails validation or is 0 bytes, it must be re-downloaded immediately and logged in the issue tracker.
4. Validation is automated using `python scripts/download_data.py --verify-checksums`.
