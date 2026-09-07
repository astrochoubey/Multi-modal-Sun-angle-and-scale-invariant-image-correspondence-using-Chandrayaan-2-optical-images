# Planetary Dataset Specification & Acquisition Guide

**Project:** Multi-modal, Sun-angle and Scale-invariant Image Correspondence using Chandrayaan-2 Optical Images  
**Role Owner:** Member D — Data, Metadata, and Planetary-Geospatial Lead  
**Audit Reviewers:** Member A (Lead) & Member B (Evaluation)  

---

## 1. Supported Sensors & Missions

The project evaluates image pairs drawn from five key planetary remote sensing instruments:

| Mission | Instrument | Type | Nominal Resolution | Spectral Coverage | Data Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Chandrayaan-2** (ISRO) | **OHRC** (Orbiter High Resolution Camera) | Panchromatic Pushbroom | **0.25 m/pixel** | 450–700 nm | PDS4 (.xml, .img) |
| **Chandrayaan-2** (ISRO) | **TMC-2** (Terrain Mapping Camera-2) | Triplet Stereo Pushbroom | **5.0 m/pixel** | 500–850 nm | PDS4 (.xml, .img) |
| **Chandrayaan-2** (ISRO) | **IIRS** (Imaging Infrared Spectrometer) | Hyperspectral Pushbroom | **$\approx 80$ m/pixel** | 0.8–5.0 $\mu\text{m}$ | PDS4 (.xml, .tab) |
| **Lunar Reconnaissance Orbiter** (NASA) | **LROC NAC** (Narrow Angle Camera) | Line-scan Pushbroom | **0.5–1.2 m/pixel** | 400–750 nm | PDS3 / PDS4 (.IMG, .xml) |
| **SELENE (Kaguya)** (JAXA) | **TC** (Terrain Camera) | Stereo Pushbroom | **10.0 m/pixel** | 430–850 nm | PDS3 / GeoTIFF |

---

## 2. Directory Layout & Data Storage Rules

In compliance with the *Team Task Playbook*, raw planetary datasets are stored locally and are strictly excluded from Git tracking:

```
data/
├── external/                # [GIT IGNORED] Raw uncompressed PDS bundles & archives
│   ├── ch2/
│   │   ├── pilot/           # Member D raw OHRC / TMC-2 pilot downloads
│   │   └── challenge/       # Locked evaluation test scenes
│   └── lro/
│       └── pilot/           # Raw LROC NAC .IMG files
├── manifests/
│   ├── pilot_manifest.csv   # Metadata table for DEV-01 through DEV-04
│   └── test_manifest.csv    # Metadata table for TEST-01 through TEST-08
├── processed/               # Extracted, normalized GeoTIFF / NumPy patches
├── patches/                 # Pre-tiled sub-swaths for LoFTR / LightGlue
└── test/
    └── ground_truth/        # Manually verified control tie-point CSV files
```

> [!CAUTION]
> Raw planetary images are multi-gigabyte files. Never stage or commit files under `data/external/` into the Git repository. Always verify `.gitignore` before committing.

---

## 3. Four-Pair Pilot Manifest Schema (`DEV-01` to `DEV-04`)

The pilot manifest file `data/manifests/pilot_manifest.csv` records the following mandatory metadata attributes for each source/reference pair:

1. `pair_id`: Unique identifier (e.g. `DEV-01`)
2. `source_product_id`: ISRO PDS4 product ID
3. `source_sensor`: OHRC / TMC2_NADIR / TMC2_FORE / TMC2_AFT / IIRS
4. `source_gsd_m`: Ground sampling distance in meters
5. `source_solar_incidence_deg`: Solar incidence angle ($i$)
6. `source_solar_azimuth_deg`: Solar azimuth angle ($\phi$)
7. `ref_product_id`: Reference product ID (LRO NAC / TMC-2)
8. `ref_sensor`: LROC_NAC / TMC2_NADIR
9. `ref_gsd_m`: Reference GSD in meters
10. `ref_solar_incidence_deg`: Reference solar incidence angle
11. `ref_solar_azimuth_deg`: Reference solar azimuth angle
12. `sun_angle_delta_deg`: Angular difference in solar illumination vectors
13. `scale_ratio`: $GSD_{ref} / GSD_{src}$
14. `source_file_path`: Relative path under `data/external/`
15. `ref_file_path`: Relative path under `data/external/`
16. `sha256_checksum`: Integrity verification hash

---

## 4. Operational Directives & Acquisition Links
- ISRO ISSDC MapBrowse Portal: [https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/)
- NASA LROC QuickMap & ACT-REACT Target Search: [https://quickmap.lroc.asu.edu/](https://quickmap.lroc.asu.edu/)
- USGS Lunar Astrogeology Planetary Data System: [https://pds-imaging.jpl.nasa.gov/](https://pds-imaging.jpl.nasa.gov/)
- Detailed handoff requirements from Member B: [docs/research/HANDOFF_SUMMARY.md](file:///Users/prachichoubey/Desktop/Projects/lunar-image-registration/docs/research/HANDOFF_SUMMARY.md)
