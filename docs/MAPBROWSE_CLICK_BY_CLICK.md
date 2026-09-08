# ISRO PRADAN & MapBrowse Click-by-Click User Guide

**Task ID:** D-01 (Part 2 & Portal Operations)  
**Author:** Member D — Data, Metadata, and Planetary-Geospatial Lead  
**Phase:** Phase 1 — Data Foundation  
**Date:** 2026-09-07  

---

## 1. Introduction

This operational guide provides step-by-step instructions for searching, filtering, inspecting, and downloading calibrated **Chandrayaan-2** datasets from the ISRO ISSDC PRADAN portal and matching reference scenes from NASA's LROC QuickMap.

---

## 2. Navigating the ISRO ISSDC PRADAN MapBrowse Portal

### Step 1: Portal Access and Authentication
1. Open a modern web browser and navigate to: **[https://pradan.issdc.gov.in/ch2/](https://pradan.issdc.gov.in/ch2/)**
2. Click **Sign In** in the upper-right corner.
3. If you do not have an active account, click **Register** to create a free planetary researcher profile with ISRO ISSDC. Verify your email and log in.

### Step 2: Selecting Mission & Instrument Payloads
1. From the top navigation bar, select **Map Search** (or **Search by Attributes**).
2. Under the **Mission** dropdown, verify that **Chandrayaan-2** is selected.
3. In the **Payload / Instrument** checklist:
   - Check **OHRC** to search for 0.25 m sub-meter high-resolution swaths.
   - Check **TMC-2** to search for 5.0 m stereo/nadir coverage.
   - *Note: Leave IIRS unchecked for the Phase 1 pilot.*

### Step 3: Setting Spatial Bounds (Lunar Footprint)
1. On the interactive 2D/3D lunar globe or map projection:
   - Use the polygon/bounding box tool to draw a search rectangle over target craters (e.g. **Boguslawsky**, **Manzinus**, **Simpelius**, or **Moretus** in the South Polar region between latitudes $65^\circ\text{S}$ and $85^\circ\text{S}$).
   - Alternatively, enter exact coordinate bounds:
     - Minimum Latitude: `-75.0`
     - Maximum Latitude: `-68.0`
     - Minimum Longitude: `30.0`
     - Maximum Longitude: `55.0`

### Step 4: Filtering by Product Level & Calibration
1. Expand the **Product Parameters** panel.
2. Under **Processing Level**, select **Level-2 (Calibrated Radiance/Reflectance)**.
   - *Do not download Raw Level-0/Level-1A telemetry, as they lack radiometric calibration and optical distortion matrices.*
3. Review the **Solar Incidence Angle** filter:
   - For benign illumination baselines (`DEV-01`), filter for $45^\circ \le i \le 65^\circ$.
   - For extreme illumination stress tests (`TEST-01`), filter for $i \ge 75^\circ$.

### Step 5: Inspecting the Product Bundle & Metadata
1. In the search results table, click the **Detail / Eye Icon** next to a product ID (e.g., `ch2_tmc_ncn_...`).
2. Verify:
   - The browse image displays clear crater topography without high sensor dropouts or clouds.
   - The metadata tab lists valid values for `Incidence Angle`, `Azimuth Angle`, and `Sub-solar Latitude/Longitude`.
3. Record the exact **Product ID**, **Product URL**, and **GSD**.

### Step 6: Downloading and Archiving
1. Add the selected item to your **Download Cart**.
2. Click **Download Bundle**. Save the resulting `.zip` / `.tar` file directly to a temporary staging folder.
3. Unpack the bundle:
   ```bash
   tar -xvf ch2_bundle.tar -C data/external/ch2/pilot/
   ```
4. Confirm that the directory contains:
   - `<product_id>.img`: The primary binary image raster.
   - `<product_id>.xml`: The PDS4 observational product XML label.
   - `<product_id>_geom.csv`: Spacecraft geometry and pixel coordinate mapping.

---

## 3. Finding Overlapping Reference Imagery on NASA LROC QuickMap

Once a Chandrayaan-2 footprint has been confirmed, use NASA's LROC QuickMap to retrieve the exact overlapping high-resolution reference swath:

### Step 1: Open LROC QuickMap
1. Navigate to: **[https://quickmap.lroc.asu.edu/](https://quickmap.lroc.asu.edu/)**
2. In the coordinate search bar in the upper center, enter the center latitude and longitude from your Chandrayaan-2 metadata (e.g., `-70.85, 42.10`).

### Step 2: Overlay Footprints and Filter NAC Frames
1. In the left-hand toolbar, click **Layers** $\rightarrow$ **Overlays** $\rightarrow$ **LROC NAC Footprints**.
2. Click the **Filter (Funnel Icon)** next to NAC Footprints:
   - **Incidence Angle:** Set within $\pm 15^\circ$ of your source image's solar incidence for baseline matching, or $> 45^\circ$ difference for challenge pairs.
   - **Resolution:** Limit to $\le 1.5\text{ m/pixel}$.
3. Click directly on the overlapping footprint on the map.

### Step 3: Inspect Product Page & Download NAC .IMG
1. In the popup window, note the **NAC Product ID** (e.g., `M1142493921RC`).
2. Click **View Product Page** to open the NASA PDS imaging archive record.
3. Download the Calibrated Data Record (**CDR**): `<product_id>.IMG` and its associated label/index files.
4. Save the file directly under:
   ```
   data/external/lro/pilot/
   ```
5. Record the LROC NAC metadata: Date, Resolution, Solar Incidence, Solar Azimuth, Emission Angle, and Phase Angle.

---

## 4. Verification Checklist Before Handing to Engineering

Before logging any pair in `pilot_manifest.csv`, Member D must verify:
- [ ] Both source and reference images cover overlapping ground terrain ($\ge 50\%$ spatial overlap).
- [ ] The raw files are non-zero bytes on disk.
- [ ] PDS4 XML label matches the binary `.img` file dimensions (lines $\times$ samples).
- [ ] SHA-256 checksum has been computed and logged.
- [ ] Neither the images nor the archives are tracked by Git (`git status` confirms untracked / ignored).
