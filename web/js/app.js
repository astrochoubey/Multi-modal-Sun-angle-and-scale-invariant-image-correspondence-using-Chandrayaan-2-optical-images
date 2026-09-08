/**
 * Lunar Image Registration Studio - Frontend Application
 * Interactivity for Split Curtain, Feature Visualizations, Telemetry, and API
 */

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentPreset = 'primary';
  let refFile = null;
  let srcFile = null;
  let lastRegistrationResult = null;

  // DOM Elements
  const runBtn = document.getElementById('run-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnIcon = document.getElementById('btn-icon');
  const btnText = document.getElementById('btn-text');

  const presetBtns = document.querySelectorAll('.preset-btn');
  const customUploadZone = document.getElementById('custom-upload-zone');
  const toggleAdvancedBtn = document.getElementById('toggle-advanced-btn');
  const advancedParamsDrawer = document.getElementById('advanced-params-drawer');
  const chevronIcon = toggleAdvancedBtn.querySelector('.chevron-icon');

  // Sliders
  const paramNFeatures = document.getElementById('param-nfeatures');
  const valNFeatures = document.getElementById('val-nfeatures');
  const paramRatio = document.getElementById('param-ratio');
  const valRatio = document.getElementById('val-ratio');
  const paramRansac = document.getElementById('param-ransac');
  const valRansac = document.getElementById('val-ransac');
  const paramClahe = document.getElementById('param-clahe');
  const valClahe = document.getElementById('val-clahe');

  // File Upload Elements
  const dropRef = document.getElementById('drop-reference');
  const dropSrc = document.getElementById('drop-source');
  const refInput = document.getElementById('ref-file-input');
  const srcInput = document.getElementById('src-file-input');
  const refChosenName = document.getElementById('ref-chosen-name');
  const srcChosenName = document.getElementById('src-chosen-name');

  // Metrics
  const metricRmse = document.getElementById('metric-rmse');
  const metricInlierRatio = document.getElementById('metric-inlier-ratio');
  const metricInliers = document.getElementById('metric-inliers');
  const metricMatches = document.getElementById('metric-matches');
  const metricKeypoints = document.getElementById('metric-keypoints');
  const metricSrcKp = document.getElementById('metric-src-kp');
  const metricRefKp = document.getElementById('metric-ref-kp');
  const metricLatency = document.getElementById('metric-latency');
  const matrixDisplay = document.getElementById('matrix-display');

  // Visualizer Images
  const imgRefBottom = document.getElementById('img-reference-bottom');
  const imgRegTop = document.getElementById('img-registered-top');
  const imgMatches = document.getElementById('img-matches');
  const imgCheckerboard = document.getElementById('img-checkerboard');
  const imgDifference = document.getElementById('img-difference');
  const imgBlendRef = document.getElementById('img-blend-ref');
  const imgBlendReg = document.getElementById('img-blend-reg');

  // Split Screen Slider Elements
  const splitWrapper = document.getElementById('split-wrapper');
  const splitOverlay = document.getElementById('split-overlay');
  const splitHandle = document.getElementById('split-handle');

  // Tabs
  const visTabs = document.querySelectorAll('.vis-tab');
  const viewPanels = document.querySelectorAll('.view-panel');

  // Alpha Blend
  const blendSlider = document.getElementById('blend-slider');
  const blendVal = document.getElementById('blend-val');

  // Download
  const downloadRegBtn = document.getElementById('download-reg-btn');

  // Toast
  const toast = document.getElementById('toast');

  // -------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------
  function showToast(message, duration = 3000) {
    toast.textContent = message;
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, duration);
  }

  // Bind Slider Value Displays
  paramNFeatures.addEventListener('input', () => valNFeatures.textContent = paramNFeatures.value);
  paramRatio.addEventListener('input', () => valRatio.textContent = parseFloat(paramRatio.value).toFixed(2));
  paramRansac.addEventListener('input', () => valRansac.textContent = parseFloat(paramRansac.value).toFixed(1));
  paramClahe.addEventListener('input', () => valClahe.textContent = parseFloat(paramClahe.value).toFixed(1));

  // Toggle Advanced Drawer
  toggleAdvancedBtn.addEventListener('click', () => {
    const isHidden = advancedParamsDrawer.style.display === 'none';
    advancedParamsDrawer.style.display = isHidden ? 'grid' : 'none';
    chevronIcon.classList.toggle('rotate', isHidden);
  });

  // Preset Selection
  presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      presetBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentPreset = btn.dataset.preset;

      if (currentPreset === 'custom') {
        customUploadZone.style.display = 'grid';
      } else {
        customUploadZone.style.display = 'none';
        // Automatically trigger registration for selected preset
        executeRegistration();
      }
    });
  });

  // File Upload Handlers
  function setupDropBox(dropBox, fileInput, nameDisplay, onSelect) {
    dropBox.addEventListener('click', () => fileInput.click());
    dropBox.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropBox.classList.add('dragover');
    });
    dropBox.addEventListener('dragleave', () => dropBox.classList.remove('dragover'));
    dropBox.addEventListener('drop', (e) => {
      e.preventDefault();
      dropBox.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        onSelect(file);
        nameDisplay.textContent = file.name;
      }
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        onSelect(file);
        nameDisplay.textContent = file.name;
      }
    });
  }

  setupDropBox(dropRef, refInput, refChosenName, (f) => (refFile = f));
  setupDropBox(dropSrc, srcInput, srcChosenName, (f) => (srcFile = f));

  // -------------------------------------------------------------
  // Tabs Navigation
  // -------------------------------------------------------------
  visTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      visTabs.forEach((t) => t.classList.remove('active'));
      viewPanels.forEach((p) => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPanel = document.getElementById(`panel-${tab.dataset.tab}`);
      if (targetPanel) {
        targetPanel.classList.add('active');
        if (tab.dataset.tab === 'split') {
          syncSplitWidth();
        }
      }
    });
  });

  // -------------------------------------------------------------
  // Interactive Split Curtain Slider
  // -------------------------------------------------------------
  let isDragging = false;

  function syncSplitWidth() {
    if (splitWrapper) {
      const containerWidth = splitWrapper.getBoundingClientRect().width;
      if (imgRegTop) {
        imgRegTop.style.width = `${containerWidth}px`;
      }
    }
  }

  window.addEventListener('resize', syncSplitWidth);

  function setSplitPosition(clientX) {
    const rect = splitWrapper.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const percent = (x / rect.width) * 100;

    splitOverlay.style.width = `${percent}%`;
    splitHandle.style.left = `${percent}%`;
  }

  splitWrapper.addEventListener('mousedown', (e) => {
    isDragging = true;
    setSplitPosition(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    setSplitPosition(e.clientX);
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  // Touch support
  splitWrapper.addEventListener('touchstart', (e) => {
    isDragging = true;
    setSplitPosition(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    setSplitPosition(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchend', () => {
    isDragging = false;
  });

  // -------------------------------------------------------------
  // Alpha Blend Slider
  // -------------------------------------------------------------
  blendSlider.addEventListener('input', () => {
    const val = blendSlider.value;
    const alpha = val / 100.0;
    imgBlendReg.style.opacity = alpha;
    blendVal.textContent = `${100 - val}% Ref / ${val}% Reg`;
  });

  // -------------------------------------------------------------
  // Pipeline Execution (API Call)
  // -------------------------------------------------------------
  async function executeRegistration() {
    // Validate custom upload
    if (currentPreset === 'custom' && (!refFile || !srcFile)) {
      showToast('⚠️ Please select both Reference and Source images.');
      return;
    }

    // Set UI loading state
    runBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    btnIcon.style.display = 'none';
    btnText.textContent = 'Registering Imagery...';

    const formData = new FormData();
    formData.append('preset_id', currentPreset);
    formData.append('n_features', paramNFeatures.value);
    formData.append('ratio_threshold', paramRatio.value);
    formData.append('reprojection_threshold', paramRansac.value);
    formData.append('clip_limit', paramClahe.value);

    if (currentPreset === 'custom') {
      formData.append('reference_file', refFile);
      formData.append('source_file', srcFile);
    }

    try {
      const response = await fetch('/api/register', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Registration failed');
      }

      lastRegistrationResult = data;

      // Update Telemetry Metrics
      const m = data.metrics;
      metricRmse.textContent = m.rmse_pixels.toFixed(4);
      metricInlierRatio.textContent = m.inlier_ratio.toFixed(2);
      metricInliers.textContent = m.inliers.toLocaleString();
      metricMatches.textContent = m.good_matches.toLocaleString();
      metricKeypoints.textContent = (m.source_keypoints + m.reference_keypoints).toLocaleString();
      metricSrcKp.textContent = m.source_keypoints.toLocaleString();
      metricRefKp.textContent = m.reference_keypoints.toLocaleString();
      metricLatency.textContent = m.latency_ms;

      // Format Homography Matrix
      if (m.homography) {
        const matrixStr = m.homography
          .map((row) => '[' + row.map((v) => v.toFixed(6).padStart(12, ' ')).join(', ') + ' ]')
          .join('\n');
        matrixDisplay.textContent = matrixStr;
      }

      // Update Visualizer Images
      const imgs = data.images;
      imgRefBottom.src = imgs.reference;
      imgRegTop.src = imgs.registered;
      imgMatches.src = imgs.matches;
      imgCheckerboard.src = imgs.checkerboard;
      imgDifference.src = imgs.difference;
      imgBlendRef.src = imgs.reference;
      imgBlendReg.src = imgs.registered;

      // Make sure curtain image width matches container after load
      imgRefBottom.onload = syncSplitWidth;
      syncSplitWidth();

      showToast(`✅ Registration complete! RMSE: ${m.rmse_pixels.toFixed(4)} px (${m.inlier_ratio}% inliers)`);
    } catch (err) {
      console.error(err);
      showToast(`❌ Error: ${err.message}`);
    } finally {
      runBtn.disabled = false;
      btnSpinner.style.display = 'none';
      btnIcon.style.display = 'inline-block';
      btnText.textContent = 'Execute Registration Pipeline';
    }
  }

  runBtn.addEventListener('click', executeRegistration);

  // -------------------------------------------------------------
  // Download Output Handler
  // -------------------------------------------------------------
  downloadRegBtn.addEventListener('click', () => {
    if (!lastRegistrationResult || !lastRegistrationResult.images.registered) {
      showToast('⚠️ Please run registration first.');
      return;
    }
    const link = document.createElement('a');
    link.href = lastRegistrationResult.images.registered;
    link.download = `chandrayaan2_registered_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('💾 Registered image downloaded.');
  });

  // Initial Run on Load
  executeRegistration();
});
