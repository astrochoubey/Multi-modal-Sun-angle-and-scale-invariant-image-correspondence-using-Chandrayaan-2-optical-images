# Research Bibliography & Source Annotations

**Task ID:** B-01 (Step 1 & Step 2)  
**Author:** Member B — Research, Novelty, and Evaluation Lead  
**Phase:** Phase 0 — Research and Requirements  
**Date:** 2026-09-07  
**Status:** Complete (14 Peer-Reviewed / Institutional Sources)  

---

## Overview

This bibliography documents peer-reviewed papers, space agency technical documentation, and foundational algorithms directly relevant to the challenge of **Multi-modal, Sun-angle, and Scale-invariant Image Registration for Chandrayaan-2 Lunar Imagery**. Every source is annotated using the rigorous 11-field schema specified in the *Team Task Playbook*.

---

## 1. Illumination & Radiation-Invariant Feature Representation

### [01] Li et al. (2020) — RIFT: Radiation-variation Insensitive Feature Transform
- **Full citation:** Li, J., Hu, Q., & Ai, M. (2020). RIFT: Multi-modal image matching based on radiation-variation insensitive feature transform. *IEEE Transactions on Image Processing*, 29, 3296–3310.
- **Link/DOI:** [https://doi.org/10.1109/TIP.2019.2959244](https://doi.org/10.1109/TIP.2019.2959244)
- **Year:** 2020
- **Problem addressed:** Multi-modal image matching where extreme radiometric disparities, nonlinear intensity distortions, and illumination reversals cause traditional gradient-based descriptors (SIFT, SURF) to fail catastrophically.
- **Data/sensors:** Optical-to-infrared, optical-to-SAR, optical-to-depth map, and multi-temporal remote sensing pairs.
- **Method:** Employs phase congruency (PC) rather than image intensity gradients. Constructs a Maximum Index Map (MIM) from log-Gabor filter responses across multiple orientations and scales to achieve invariance to monotonic and non-monotonic radiation variations.
- **Metrics:** Number of Correct Matches (NCM), Correct Matching Rate (CMR / inlier ratio), Root Mean Square Error (RMSE) on tie points.
- **Strengths:** Outstanding robustness to non-linear intensity changes and shadow reversals; does not require deep learning models or pre-training on satellite imagery; rotation-invariant via circular sequence convolution.
- **Limitations:** Computationally expensive due to multi-scale, multi-orientation log-Gabor bank convolutions; sensitive to large scale differences (> 3x) without explicit scale-space pyramids.
- **What we can reuse:** Log-Gabor filter bank formulation, Phase Congruency (PC) calculation routines, and Maximum Index Map (MIM) feature descriptors for optical-to-infrared (TMC-2 to IIRS) and extreme Sun-angle pairs.
- **Why it does not completely solve our problem:** Lacks native multi-scale pyramid support to bridge large scale disparities (e.g., OHRC 0.25 m/pixel vs. TMC-2 5.0 m/pixel = 20x scale jump) and does not model planetary pushbroom epipolar geometry.

---

### [02] Kovesi (1999, 2000) — Phase Congruency for Feature Detection
- **Full citation:** Kovesi, P. (1999). Image features from phase congruency. *Videre: Journal of Computer Vision Research*, 1(3), 1–26; Kovesi, P. (2000). Phase congruency detects corners and edges. *Australian Pattern Recognition Society Conference (DICTA 2000)*, 309–318.
- **Link/DOI:** [https://www.peterkovesi.com/projects/academic/#phasecongruency](https://www.peterkovesi.com/projects/academic/#phasecongruency)
- **Year:** 1999 / 2000
- **Problem addressed:** Feature detection (edges and corners) invariant to illumination changes, shadow gradients, and image magnification/contrast variations.
- **Data/sensors:** Synthetic patterns, natural images, and medical imaging.
- **Method:** Postulates that human visual perception marks features at spatial points where the Fourier components are maximally in phase, rather than where gradient magnitudes peak. Uses 2D log-Gabor wavelets to estimate local energy and amplitude across multiple frequency scales.
- **Metrics:** Feature localization accuracy, repeatability under contrast attenuation, noise immunity.
- **Strengths:** Completely invariant to affine illumination changes and contrast stretching; responds equally well to step edges, roof edges, and crater rims regardless of lighting direction.
- **Limitations:** High spatial complexity; sensitive to high-frequency speckle noise if filter frequency bands are untuned; computationally slower than OpenCV gradient filters.
- **What we can reuse:** Algorithmic foundation for structural edge and corner extraction on lunar images where crater shadows flip completely between low-Sun and high-Sun observations.
- **Why it does not completely solve our problem:** Provides a scalar feature map or corner response; it is not an end-to-end descriptor matcher or geometric estimator.

---

### [03] Xiang et al. (2018) — OS-SIFT for Optical-to-SAR Registration
- **Full citation:** Xiang, Y., Wang, F., & You, H. (2018). OS-SIFT: A robust SIFT-like algorithm for high-resolution optical-to-SAR image registration in urban areas. *IEEE Transactions on Geoscience and Remote Sensing*, 56(6), 3078–3090.
- **Link/DOI:** [https://doi.org/10.1109/TGRS.2017.2782042](https://doi.org/10.1109/TGRS.2017.2782042)
- **Year:** 2018
- **Problem addressed:** Registration of optical and synthetic aperture radar (SAR) images with severe speckle noise and distinct imaging geometries.
- **Data/sensors:** TerraSAR-X, Gaofen-2, aerial high-resolution optical and SAR imagery.
- **Method:** Defines consistent gradient operators (Ratio of Exponentially Weighted Averages, ROEWA, and Sobel) to generate consistent gradient magnitude and orientation representations across both sensors, followed by an augmented SIFT-like histogram.
- **Metrics:** Matching success rate, number of correct matches, registration error (pixels).
- **Strengths:** Effectively bridges the structural gap between scattering-based and reflectance-based remote sensing modalities.
- **Limitations:** Tuned specifically for SAR speckle statistics; performance drops under extreme topography and scale disparity.
- **What we can reuse:** Concept of ratio-based gradient operators for handling sensor modality discrepancies between Chandrayaan-2 TMC-2 (optical reflectance) and IIRS (infrared absorption/emission).
- **Why it does not completely solve our problem:** Lunar topography under changing solar incidence introduces structural shadow displacement, not just diffuse/specular ratio variance.

---

## 2. Classical Sparse Feature Detection & Matching

### [04] Lowe (2004) — Scale-Invariant Feature Transform (SIFT)
- **Full citation:** Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110.
- **Link/DOI:** [https://doi.org/10.1023/B:VISI.0000029664.99615.94](https://doi.org/10.1023/B:VISI.0000029664.99615.94)
- **Year:** 2004
- **Problem addressed:** Keypoint detection and description invariant to image scaling, rotation, 3D viewpoint changes, and additive illumination changes.
- **Data/sensors:** Standard optical benchmark imagery.
- **Method:** Scale-space extrema detection via Difference-of-Gaussians (DoG), sub-pixel keypoint localization, orientation assignment from local image gradients, and 128-dimensional normalized gradient orientation histograms. Matches filtered using the second-nearest-neighbor distance ratio test ($d_1 / d_2 \le 0.75 - 0.80$).
- **Metrics:** Repeatability rate, matching precision, percentage of correct keypoint matches under rotation and scale changes.
- **Strengths:** Highly reliable baseline, mathematically transparent, robust to moderate scale changes (up to ~2.5x) and in-plane rotations.
- **Limitations:** Descriptor relies on raw intensity gradients; fails when illumination direction alters shadow casting directions (e.g. crater rim shadows flip 180° when solar azimuth changes); high failure rate at scale ratios > 3x.
- **What we can reuse:** Standard OpenCV implementation (`cv2.SIFT_create`) as the baseline benchmark method against which all novel pipelines are evaluated.
- **Why it does not completely solve our problem:** As shown in our synthetic baseline audit (`docs/engineering/BASELINE_AUDIT.md`), SIFT matches degrade drastically when shadows reverse or when optical/infrared cross-modal textures differ.

---

### [05] Morel & Yu (2009) — ASIFT: A New Framework for Fully Affine Invariant Image Comparison
- **Full citation:** Morel, J. M., & Yu, G. (2009). ASIFT: A new framework for fully affine invariant image comparison. *SIAM Journal on Imaging Sciences*, 2(2), 438–469.
- **Link/DOI:** [https://doi.org/10.1137/080732730](https://doi.org/10.1137/080732730)
- **Year:** 2009
- **Problem addressed:** Matching images subject to severe out-of-plane tilt and viewpoint variation where standard SIFT fails.
- **Data/sensors:** Oblique aerial photography, architectural views.
- **Method:** Simulates all possible affine camera distortions by varying camera latitude (tilt) and longitude (rotation) angles across a discrete parameter space, executing SIFT on simulated images.
- **Metrics:** Number of matches, tilt coverage, precision.
- **Strengths:** Truly invariant to extreme affine perspective distortions up to tilt transitions of $\tau \approx 6$.
- **Limitations:** Extremely high computational overhead (orders of magnitude slower than SIFT); does not address non-monotonic illumination or shadow inversion.
- **What we can reuse:** Affine view synthesis techniques for handling oblique off-nadir observations (e.g. TMC-2 Fore/Aft stereo angles vs. OHRC Nadir).
- **Why it does not completely solve our problem:** Computational complexity is prohibitive for large planetary swaths, and it cannot resolve lighting shifts where topography shadows migrate.

---

## 3. Deep Learned Keypoint & Dense Matching

### [06] DeTone, Malisiewicz, & Rabinovich (2018) — SuperPoint: Self-Supervised Interest Point Detection and Description
- **Full citation:** DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-supervised interest point detection and description for versatile computer vision tasks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*, 224–236.
- **Link/DOI:** [https://arxiv.org/abs/1712.07629](https://arxiv.org/abs/1712.07629)
- **Year:** 2018
- **Problem addressed:** Joint interest point detection and descriptor extraction trained self-supervised on homographic transformations.
- **Data/sensors:** Synthetic geometric shapes (Synthetic Shapes) followed by MS-COCO natural images via Homographic Adaptation.
- **Method:** Fully convolutional architecture operating at full image resolution. Computes semi-dense 2D keypoint heatmap and 256-dimensional $L_2$-normalized descriptors via a shared VGG-style backbone encoder with sub-pixel soft argmax.
- **Metrics:** Repeatability, Homography estimation accuracy ($\epsilon \le 3$ px), Mean Average Precision (mAP).
- **Strengths:** Highly repeatable corner and salient structure detection; robust to photometric distortions; lightweight and fast inference on GPU (~15 ms for 640x480).
- **Limitations:** Pre-trained weights are biased toward terrestrial man-made architectural edges and corners; uncalibrated response to soft crater rims and self-shadowed lunar regolith without fine-tuning.
- **What we can reuse:** SuperPoint keypoint detector and descriptor extractor as a modern sparse learned baseline.
- **Why it does not completely solve our problem:** Domain gap: terrestrial training data has distinct high-frequency textures and lighting models compared to lunar Hapke scattering and crater morphology.

---

### [07] Sarlin, DeTone, Malisiewicz, & Rabinovich (2020) — SuperGlue: Learning Feature Matching with Graph Neural Networks
- **Full citation:** Sarlin, P. E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). SuperGlue: Learning feature matching with graph neural networks. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 4938–4947.
- **Link/DOI:** [https://arxiv.org/abs/1911.11763](https://arxiv.org/abs/1911.11763)
- **Year:** 2020
- **Problem addressed:** Matching feature sets under wide baselines and extreme visual appearance differences, replacing heuristic nearest-neighbor matching and ratio tests.
- **Data/sensors:** ScanNet (indoor) and MegaDepth (outdoor terrestrial).
- **Method:** Attentional Graph Neural Network (GNN) combining self-attention (within image) and cross-attention (between images), solving optimal transport via differentiable Sinkhorn algorithm with a dustbin for unmatched points.
- **Metrics:** Pose estimation AUC at $5^\circ, 10^\circ, 20^\circ$, Matching precision and recall.
- **Strengths:** Exceptional outlier rejection and structural context reasoning; capable of finding correspondences even when local patches are visually degraded.
- **Limitations:** High memory consumption; proprietary non-commercial research-only license from Magic Leap; quadratic complexity $\mathcal{O}(N_1 N_2)$ with keypoint counts.
- **What we can reuse:** Theoretical formulation of graph-attentional feature contextualization and optimal transport matching.
- **Why it does not completely solve our problem:** Strict non-commercial licensing constraints prevent open planetary pipeline distribution; computational footprint limits gigapixel satellite swath processing.

---

### [08] Lindenberger, Sarlin, & Pollefeys (2023) — LightGlue: Local Feature Matching at Light Speed
- **Full citation:** Lindenberger, P., Sarlin, P. E., & Pollefeys, M. (2023). LightGlue: Local feature matching at light speed. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 17845–17855.
- **Link/DOI:** [https://arxiv.org/abs/2306.13643](https://arxiv.org/abs/2306.13643)
- **Year:** 2023
- **Problem addressed:** High latency, fixed depth, and memory bottlenecks of SuperGlue while retaining attentional correspondence power.
- **Data/sensors:** MegaDepth, ScanNet, and synthetic homographies.
- **Method:** Adaptive transformer architecture that introspects match difficulty per layer. Employs early exit mechanisms for easily matched pairs, lightweight attention heads, and efficient dual-softmax assignment. Permissive Apache 2.0 license.
- **Metrics:** Relative pose accuracy AUC, Matching speed (FPS), VRAM consumption, match precision.
- **Strengths:** Permissive open-source license (Apache 2.0); 3x to 5x faster than SuperGlue; lower memory overhead; compatible with multiple front-ends (SuperPoint, DISK, ALIKED, SIFT).
- **Limitations:** Still requires pre-extracted keypoints; domain transfer to lunar surface depends on front-end keypoint stability.
- **What we can reuse:** Primary candidate for Phase 4 learned matching module in `src/lunar_registration/matching/` paired with SuperPoint or SIFT keypoints.
- **Why it does not completely solve our problem:** Requires GPU hardware for real-time throughput; performance still degrades when scale disparity exceeds 4x without multi-scale image tiling.

---

### [09] Sun, Shen, Yuan, Zhou, Bao, & Zhou (2021) — LoFTR: Detector-Free Local Feature Matching with Transformers
- **Full citation:** Sun, J., Shen, Z., Yuan, Y., Zhou, W., Bao, H., & Zhou, X. (2021). LoFTR: Detector-free local feature matching with transformers. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 8922–8931.
- **Link/DOI:** [https://arxiv.org/abs/2104.00680](https://arxiv.org/abs/2104.00680)
- **Year:** 2021
- **Problem addressed:** Keypoint detection failure in textureless, repetitive, or poorly illuminated regions where interest point detectors extract few or no repeatable keypoints.
- **Data/sensors:** MegaDepth (outdoor) and ScanNet (indoor).
- **Method:** Detector-free correspondence. Extracts feature maps at coarse ($1/8$) and fine ($1/2$) resolutions using a CNN backbone. Applies Linear Transformers with self and cross-attention at coarse scale, finds coarse matches via mutual nearest neighbors, and refines them to sub-pixel coordinates using a correlation window at fine scale.
- **Metrics:** Homography accuracy ($< 1, 3, 5$ pixels), Pose estimation AUC, inlier ratio.
- **Strengths:** Thrives in low-texture regions (e.g. flat lunar mare plains); delivers dense, uniformly distributed correspondence grids; does not rely on fragile feature point corner detection.
- **Limitations:** Extremely high VRAM footprint (requires 8+ GB VRAM for images $> 800 \times 800$); rigid image dimension constraints; fine-scale refinement can drift if global scale difference is large (> 3x).
- **What we can reuse:** Dense matcher candidate for low-contrast lunar mare surfaces where SIFT/SuperPoint detect zero features.
- **Why it does not completely solve our problem:** Full Chandrayaan-2/LRO scenes are typically $10,000 \times 4,000$ to $50,000 \times 4,000$ pixels; monolithic LoFTR cannot be directly executed on full swaths without intelligent patch-tiling and coarse geospatial indexing.

---

## 4. Robust Geometric Estimation & Spatial Outlier Rejection

### [10] Barath, Noskova, Ivashechkin, & Matas (2020) — MAGSAC++: A Fast, Reliable and Accurate Robust Estimator
- **Full citation:** Barath, D., Noskova, J., Ivashechkin, M., & Matas, J. (2020). MAGSAC++, a fast, reliable and accurate robust estimator. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 1304–1312.
- **Link/DOI:** [https://arxiv.org/abs/1912.05909](https://arxiv.org/abs/1912.05909)
- **Year:** 2020
- **Problem addressed:** Traditional RANSAC requires a user-specified hard inlier-noise threshold $\sigma$; wrong threshold choices lead to inclusion of severe outliers or elimination of valid correspondences.
- **Data/sensors:** Strecha, PhotoTourism, and KITTI benchmarks.
- **Method:** Marginalizes the inlier residual over an infinite range of noise thresholds using a $\sigma$-consensus framework, computing model quality via a closed-form likelihood integral. Implemented natively in OpenCV (`cv2.USAC_MAGSAC`).
- **Metrics:** Geometric error (RMSE), run-time, number of model evaluations, precision.
- **Strengths:** Eliminates arbitrary heuristic inlier thresholding; superior accuracy on near-planar and affine scenes; orders of magnitude faster than standard RANSAC when combined with USAC framework.
- **Limitations:** Assumes Gaussian residual distributions; complex non-planar relief on crater rims can violate single homography assumptions.
- **What we can reuse:** Direct drop-in replacement for `cv2.RANSAC` in `src/lunar_registration/geometry/homography.py` via `cv2.findHomography(..., method=cv2.USAC_MAGSAC)`.
- **Why it does not completely solve our problem:** Planar homography assumes either a flat planar surface or a purely rotating camera; lunar terrain with deep crater depth (e.g., 2–4 km rims) requires piecewise/local planar or epipolar models for large swaths.

---

### [11] Raguram, Chum, Pollefeys, Sivic, & Matas (2013) — USAC: A Universal Framework for Random Sample Consensus
- **Full citation:** Raguram, R., Chum, O., Pollefeys, M., Sivic, J., & Matas, J. (2013). USAC: A universal framework for random sample consensus. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(8), 2022–2038.
- **Link/DOI:** [https://doi.org/10.1109/TPAMI.2012.257](https://doi.org/10.1109/TPAMI.2012.257)
- **Year:** 2013
- **Problem addressed:** Inefficiency, randomness, and sub-optimal precision in classical RANSAC implementations.
- **Data/sensors:** Standard photogrammetric and two-view vision benchmarks.
- **Method:** Modular architecture integrating PROSAC (progressive sampling), $T_{d,d}$ test (fast preemptive verification), SPRT (sequential probability ratio test), and local optimization (LO-RANSAC).
- **Metrics:** Convergence speed, number of samples drawn, inlier precision, failure rate.
- **Strengths:** State-of-the-art engineering integration; speeds up consensus search by 10x–50x while guaranteeing optimal inlier detection.
- **Limitations:** Configurable parameters (SPRT thresholds, sampling weights) must be tuned to remote sensing inlier ratios.
- **What we can reuse:** Parameterized backend for all robust geometry solvers in `src/lunar_registration/geometry/`.
- **Why it does not completely solve our problem:** A robust estimator cannot recover geometry if the upstream matching stage produces 100% false correspondences due to complete shadow inversion.

---

## 5. Sub-Pixel Registration & Precision Refinement

### [12] Guizar-Sicairos, Thurman, & Fienup (2008) — Efficient Subpixel Image Registration Algorithms
- **Full citation:** Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008). Efficient subpixel image registration algorithms. *Optics Letters*, 33(2), 156–158.
- **Link/DOI:** [https://doi.org/10.1364/OL.33.000156](https://doi.org/10.1364/OL.33.000156)
- **Year:** 2008
- **Problem addressed:** Sub-pixel image translation registration without the enormous memory and computational cost of zero-padded Discrete Fourier Transforms (DFT).
- **Data/sensors:** Microscopic, astronomical, and optical imaging datasets.
- **Method:** Computes initial 2D cross-correlation via standard FFT, followed by a matrix-multiply discrete Fourier transform (matrix-multiply DFT) localized only to a small neighborhood around the peak correlation pixel.
- **Metrics:** Sub-pixel shift accuracy down to $1/20$th to $1/100$th of a pixel, computational complexity, memory usage.
- **Strengths:** Exact mathematical equivalence to high-dimensional upsampled FFT; extremely fast ($\mathcal{O}(N)$ rather than $\mathcal{O}(N \log N)$ over upsampled grids); analytically rigorous uncertainty bounds.
- **Limitations:** Purely translational model; cannot directly estimate rotation, shear, or perspective warping across full frames without patch-based decomposition.
- **What we can reuse:** Patch-level sub-pixel refinement engine in `src/lunar_registration/geometry/refinement.py` for fine-tuning candidate tie points after global homography warping.
- **Why it does not completely solve our problem:** Global transformation between multi-temporal lunar images involves rotation, perspective scale differences, and topography parallax.

---

## 6. Planetary Remote Sensing & Lunar Surface Matching

### [13] Wu et al. (2020) & Ding et al. (2021) — Lunar Photometric Normalization & Crater Matching
- **Full citation:** Wu, B., Li, F., & Ye, H. (2020). Photometric processing and seamless mosaicking of Chang'e-2 and Chang'e-4 lunar imagery. *Earth and Space Science*, 7(11), e2020EA001284; Ding, C., et al. (2021). Automated crater recognition and registration for planetary landing navigation using deep learning. *IEEE Transactions on Aerospace and Electronic Systems*, 57(4), 2311–2324.
- **Link/DOI:** [https://doi.org/10.1029/2020EA001284](https://doi.org/10.1029/2020EA001284)
- **Year:** 2020 / 2021
- **Problem addressed:** Geometric co-registration and radiometric seam eradication on lunar surfaces subjected to varying incidence, emission, and phase angles across planetary flybys and descent cameras.
- **Data/sensors:** Chang'e-2/4/5 lander descent imagery, Lunar Reconnaissance Orbiter Camera (LROC) Wide Angle/Narrow Angle Cameras.
- **Method:** Implements Lommel-Seeliger and Lunar-Lambert empirical photometric correction models using local digital elevation models (DEMs), coupled with circular crater boundary detection for tie-point formulation.
- **Metrics:** Radiometric consistency, photometric RMSE, crater center localization residual (pixels).
- **Strengths:** Addresses the core physical cause of visual discrepancy on planetary surfaces (Hapke and Lambertian scattering dynamics).
- **Limitations:** Requires accurate 3D DEMs and exact SPICE geometry kernels (spacecraft ephemeris, solar vectors); fails if high-resolution DEM is unavailable.
- **What we can reuse:** Mathematical understanding of solar incidence angles and justification for why contrast-normalization (CLAHE/Wall-filter) must substitute for full photometric correction when DEMs are absent.
- **Why it does not completely solve our problem:** When high-resolution topographic DEMs are not available a priori for newly imaged South Pole targets, full 3D photometric rendering cannot be computed.

---

### [14] Chowdhury et al. (2020) / ISRO SAC Documentation — Chandrayaan-2 Planetary Imaging Payload Specifications
- **Full citation:** Chowdhury, A. R., et al. (2020). High resolution imaging by OHRC onboard Chandrayaan-2 orbiter. *Current Science*, 118(4), 546–555; Amitabh, et al. (2021). Chandrayaan-2 TMC-2 data processing and high resolution DEM generation. *ISRO Space Applications Centre (SAC) Technical Report*, SAC-ISRO-TR-2021-02.
- **Link/DOI:** [https://www.isro.gov.in/chandrayaan2_payloads.html](https://www.isro.gov.in/chandrayaan2_payloads.html)
- **Year:** 2020 / 2021
- **Problem addressed:** Characterization of the lunar surface at sub-meter scales (0.25 m/pixel) for hazard detection, safe landing site certification, and global 3D surface reconstruction.
- **Data/sensors:** Chandrayaan-2 Orbiter High Resolution Camera (OHRC), Terrain Mapping Camera-2 (TMC-2), Imaging Infrared Spectrometer (IIRS).
- **Method:** Linear TDI (Time Delay Integration) CCD sensor in line-scan pushbroom mode; steerable roll maneuvers up to $\pm 32^\circ$ for illumination optimization.
- **Metrics:** Ground Sampling Distance (GSD), Signal-to-Noise Ratio (SNR), Modulation Transfer Function (MTF), Swath Width.
- **Strengths:** Authoritative source on sensor geometry, spectral passbands, optical distortions, and raw metadata formats (PDS4 XML/IMG, auxiliary geometry CSV).
- **Limitations:** Documents payload specifications; does not provide automated multi-modal registration algorithms.
- **What we can reuse:** Sensor physical specifications (pixel pitch, focal length, swath width, resolution ratios) to establish exact constraints in `configs/default.yaml` and pilot manifests.
- **Why it does not completely solve our problem:** It defines the hardware and data formats, but leaves automated multi-modal, sun-angle, and scale-invariant correspondence as an open research and engineering challenge.

---

## Summary Matrix of Topics Covered

| Research Area | Source Citations | Key Insight for Project |
| :--- | :--- | :--- |
| **Illumination / Radiation Invariance** | Li et al. (2020), Kovesi (1999, 2000), Xiang et al. (2018) | Phase congruency and gradient-ratio operators bypass shadow inversions and nonlinear intensity shifts. |
| **Classical Sparse Matching** | Lowe (2004), Morel & Yu (2009) | SIFT provides transparent baseline; affine synthesis handles perspective view tilt but lacks shadow invariance. |
| **Learned Matching** | DeTone et al. (2018), Sarlin et al. (2020), Lindenberger et al. (2023), Sun et al. (2021) | LightGlue offers the optimal balance of license, speed, and accuracy; LoFTR rescues textureless mare basins. |
| **Robust Geometry** | Barath et al. (2020), Raguram et al. (2013) | MAGSAC++ eliminates heuristic inlier thresholds; USAC framework accelerates consensus search. |
| **Sub-Pixel Precision** | Guizar-Sicairos et al. (2008) | Localized matrix-multiply DFT allows micro-pixel refinement around established tie points. |
| **Planetary Ground Truth & Data** | Wu et al. (2020), Ding et al. (2021), Chowdhury et al. (2020) | Hapke scattering explains why classical descriptors fail; Chandrayaan-2 sensor specs define our operational scales. |
