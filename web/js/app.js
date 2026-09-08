/**
 * Chandrayaan-2 Lunar Optical Registration Console - Frontend Application
 * Aerospace Mission Controller Interactive Architecture
 * Integrates: Floating-UI, AOS, Lucide, Web Audio Synthesizer, Dynamic Starfield
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize External Libraries
  if (window.lucide) {
    window.lucide.createIcons();
  }
  if (window.AOS) {
    window.AOS.init({
      duration: 650,
      once: true,
      easing: 'ease-out-cubic',
    });
  }

  // -------------------------------------------------------------
  // Web Audio API Sci-Fi Telemetry Sound Effects Synthesizer
  // -------------------------------------------------------------
  let audioEnabled = false;
  let audioCtx = null;
  const audioToggleBtn = document.getElementById('audio-toggle-btn');
  const audioIcon = document.getElementById('audio-icon');

  function initAudio() {
    if (!audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        audioCtx = new AudioContext();
      }
    }
  }

  function playChirp(freq = 880, type = 'sine', duration = 0.08) {
    if (!audioEnabled || !audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(freq * 1.5, audioCtx.currentTime + duration);
      gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      // Audio autoplay policy fallback
    }
  }

  function playSuccessPing() {
    if (!audioEnabled || !audioCtx) return;
    try {
      const notes = [587.33, 880, 1174.66]; // D5, A5, D6 chord
      notes.forEach((freq, idx) => {
        setTimeout(() => playChirp(freq, 'triangle', 0.18), idx * 70);
      });
    } catch (e) {}
  }

  if (audioToggleBtn) {
    audioToggleBtn.addEventListener('click', () => {
      initAudio();
      audioEnabled = !audioEnabled;
      audioToggleBtn.classList.toggle('active', audioEnabled);
      if (audioEnabled) {
        playChirp(1200, 'sine', 0.1);
        showToast('🔊 Mission Audio FX: ENABLED');
      } else {
        showToast('🔇 Mission Audio FX: MUTED');
      }
    });
  }

  // -------------------------------------------------------------
  // Live Orbital UTC Mission Clock
  // -------------------------------------------------------------
  const missionClockEl = document.getElementById('mission-clock');
  function updateMissionClock() {
    if (missionClockEl) {
      const now = new Date();
      const utcStr = now.toISOString().replace('T', ' // ').substring(0, 22) + ' UTC';
      missionClockEl.textContent = utcStr;
    }
  }
  updateMissionClock();
  setInterval(updateMissionClock, 1000);

  // -------------------------------------------------------------
  // Interactive Deep Space Starfield Background Canvas
  // -------------------------------------------------------------
  const starCanvas = document.getElementById('starfield-canvas');
  if (starCanvas) {
    const ctx = starCanvas.getContext('2d');
    let width = (starCanvas.width = window.innerWidth);
    let height = (starCanvas.height = window.innerHeight);
    const stars = [];
    const numStars = Math.min(220, Math.floor((width * height) / 8000));

    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 1.4 + 0.3,
        alpha: Math.random() * 0.7 + 0.2,
        twinkleSpeed: Math.random() * 0.02 + 0.005,
        speedX: (Math.random() - 0.5) * 0.12,
        speedY: (Math.random() - 0.5) * 0.12,
      });
    }

    let mouseX = width / 2;
    let mouseY = height / 2;
    let targetMouseX = mouseX;
    let targetMouseY = mouseY;

    window.addEventListener('mousemove', (e) => {
      targetMouseX = e.clientX;
      targetMouseY = e.clientY;
    });

    window.addEventListener('resize', () => {
      width = starCanvas.width = window.innerWidth;
      height = starCanvas.height = window.innerHeight;
    });

    function renderStarfield() {
      mouseX += (targetMouseX - mouseX) * 0.05;
      mouseY += (targetMouseY - mouseY) * 0.05;

      ctx.clearRect(0, 0, width, height);

      const offsetX = (mouseX / width - 0.5) * 20;
      const offsetY = (mouseY / height - 0.5) * 20;

      for (let i = 0; i < stars.length; i++) {
        const s = stars[i];
        s.alpha += s.twinkleSpeed;
        if (s.alpha > 0.95 || s.alpha < 0.2) {
          s.twinkleSpeed = -s.twinkleSpeed;
        }

        s.x += s.speedX;
        s.y += s.speedY;
        if (s.x < 0) s.x = width;
        if (s.x > width) s.x = 0;
        if (s.y < 0) s.y = height;
        if (s.y > height) s.y = 0;

        ctx.beginPath();
        ctx.arc(s.x + offsetX * s.radius, s.y + offsetY * s.radius, s.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(186, 230, 253, ${s.alpha})`;
        ctx.shadowBlur = s.radius > 1.2 ? 6 : 0;
        ctx.shadowColor = '#38bdf8';
        ctx.fill();
      }

      requestAnimationFrame(renderStarfield);
    }
    renderStarfield();
  }

  // -------------------------------------------------------------
  // Floating UI Tooltip System
  // -------------------------------------------------------------
  const floatingTooltip = document.getElementById('global-floating-tooltip');
  if (floatingTooltip && window.FloatingUIDOM) {
    const { computePosition, offset, shift, flip } = window.FloatingUIDOM;
    let currentRef = null;

    document.querySelectorAll('[data-tooltip]').forEach((el) => {
      el.addEventListener('mouseenter', () => {
        const text = el.getAttribute('data-tooltip');
        if (!text) return;
        currentRef = el;
        floatingTooltip.textContent = text;
        floatingTooltip.style.display = 'block';

        computePosition(el, floatingTooltip, {
          placement: 'top',
          middleware: [offset(10), flip(), shift({ padding: 8 })],
        }).then(({ x, y }) => {
          floatingTooltip.style.left = `${x}px`;
          floatingTooltip.style.top = `${y}px`;
        });
      });

      el.addEventListener('mouseleave', () => {
        if (currentRef === el) {
          floatingTooltip.style.display = 'none';
          currentRef = null;
        }
      });
    });
  }

  // -------------------------------------------------------------
  // State
  // -------------------------------------------------------------
  let currentPreset = 'primary';
  let refFile = null;
  let srcFile = null;
  let lastRegistrationResult = null;

  // DOM Elements
  const runBtn = document.getElementById('run-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnIcon = document.getElementById('btn-icon');
  const btnText = document.getElementById('btn-text');

  const presetBtns = document.querySelectorAll('.preset-card');
  const customUploadZone = document.getElementById('custom-upload-zone');
  const toggleAdvancedBtn = document.getElementById('toggle-advanced-btn');
  const advancedParamsDrawer = document.getElementById('advanced-params-drawer');
  const chevronIcon = toggleAdvancedBtn ? toggleAdvancedBtn.querySelector('.chevron-icon') : null;

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
  const copyMatrixBtn = document.getElementById('copy-matrix-btn');

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

  // Surface Telemetry HUD
  const hudCoordX = document.getElementById('hud-coord-x');
  const hudCoordY = document.getElementById('hud-coord-y');
  const hudCoordVal = document.getElementById('hud-coord-val');

  // -------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------
  function showToast(message, duration = 3200) {
    if (!toast) return;
    toast.textContent = message;
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, duration);
  }

  // Bind Slider Value Displays
  if (paramNFeatures && valNFeatures) {
    paramNFeatures.addEventListener('input', () => {
      valNFeatures.textContent = paramNFeatures.value;
      playChirp(600, 'sine', 0.02);
    });
  }
  if (paramRatio && valRatio) {
    paramRatio.addEventListener('input', () => {
      valRatio.textContent = parseFloat(paramRatio.value).toFixed(2);
      playChirp(650, 'sine', 0.02);
    });
  }
  if (paramRansac && valRansac) {
    paramRansac.addEventListener('input', () => {
      valRansac.textContent = parseFloat(paramRansac.value).toFixed(1);
      playChirp(700, 'sine', 0.02);
    });
  }
  if (paramClahe && valClahe) {
    paramClahe.addEventListener('input', () => {
      valClahe.textContent = parseFloat(paramClahe.value).toFixed(1);
      playChirp(750, 'sine', 0.02);
    });
  }

  // Toggle Advanced Drawer
  if (toggleAdvancedBtn && advancedParamsDrawer) {
    toggleAdvancedBtn.addEventListener('click', () => {
      playChirp(900, 'triangle', 0.05);
      const isHidden = advancedParamsDrawer.style.display === 'none';
      advancedParamsDrawer.style.display = isHidden ? 'grid' : 'none';
      if (chevronIcon) {
        chevronIcon.classList.toggle('rotate', isHidden);
      }
    });
  }

  // Preset Selection
  presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      playChirp(840, 'triangle', 0.06);
      presetBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentPreset = btn.dataset.preset;

      if (currentPreset === 'custom') {
        if (customUploadZone) customUploadZone.style.display = 'grid';
      } else {
        if (customUploadZone) customUploadZone.style.display = 'none';
        executeRegistration();
      }
    });
  });

  // File Upload Handlers
  function setupDropBox(dropBox, fileInput, nameDisplay, onSelect) {
    if (!dropBox || !fileInput) return;
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
        if (nameDisplay) nameDisplay.textContent = file.name;
        playChirp(950, 'sine', 0.08);
      }
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        onSelect(file);
        if (nameDisplay) nameDisplay.textContent = file.name;
        playChirp(950, 'sine', 0.08);
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
      playChirp(1000, 'sine', 0.04);
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
    if (!splitWrapper) return;
    const rect = splitWrapper.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const percent = (x / rect.width) * 100;

    if (splitOverlay) splitOverlay.style.width = `${percent}%`;
    if (splitHandle) splitHandle.style.left = `${percent}%`;
  }

  if (splitWrapper) {
    splitWrapper.addEventListener('mousedown', (e) => {
      isDragging = true;
      setSplitPosition(e.clientX);
    });

    // Surface live coordinate HUD tracking
    splitWrapper.addEventListener('mousemove', (e) => {
      const rect = splitWrapper.getBoundingClientRect();
      const px = Math.floor(e.clientX - rect.left);
      const py = Math.floor(e.clientY - rect.top);
      if (hudCoordX) hudCoordX.textContent = String(px).padStart(3, '0');
      if (hudCoordY) hudCoordY.textContent = String(py).padStart(3, '0');
      if (hudCoordVal) {
        const val = (0.7 + (Math.sin(px * 0.02) * Math.cos(py * 0.02) * 0.25)).toFixed(2);
        hudCoordVal.textContent = val;
      }
      if (isDragging) {
        setSplitPosition(e.clientX);
      }
    });

    // Touch support
    splitWrapper.addEventListener('touchstart', (e) => {
      isDragging = true;
      setSplitPosition(e.touches[0].clientX);
    }, { passive: true });

    splitWrapper.addEventListener('touchmove', (e) => {
      if (!isDragging) return;
      setSplitPosition(e.touches[0].clientX);
    }, { passive: true });
  }

  window.addEventListener('mouseup', () => (isDragging = false));
  window.addEventListener('touchend', () => (isDragging = false));

  // -------------------------------------------------------------
  // Alpha Blend Slider
  // -------------------------------------------------------------
  if (blendSlider) {
    blendSlider.addEventListener('input', () => {
      const val = blendSlider.value;
      const alpha = val / 100.0;
      if (imgBlendReg) imgBlendReg.style.opacity = alpha;
      if (blendVal) blendVal.textContent = `${100 - val}% Ref / ${val}% Reg`;
    });
  }

  // -------------------------------------------------------------
  // Copy Homography Matrix
  // -------------------------------------------------------------
  if (copyMatrixBtn && matrixDisplay) {
    copyMatrixBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      playChirp(1100, 'triangle', 0.05);
      navigator.clipboard.writeText(matrixDisplay.textContent).then(() => {
        showToast('📋 Homography Matrix copied to clipboard!');
      }).catch(() => {
        showToast('⚠️ Unable to copy matrix.');
      });
    });
  }

  // -------------------------------------------------------------
  // Pipeline Execution (API Call)
  // -------------------------------------------------------------
  async function executeRegistration() {
    playChirp(700, 'sawtooth', 0.1);

    // Validate custom upload
    if (currentPreset === 'custom' && (!refFile || !srcFile)) {
      showToast('⚠️ Incomplete Ingest: Please select both Reference and Source frames.');
      return;
    }

    // Set UI loading state
    if (runBtn) runBtn.disabled = true;
    if (btnSpinner) btnSpinner.style.display = 'inline-block';
    if (btnIcon) btnIcon.style.display = 'none';
    if (btnText) btnText.textContent = 'Registering Optical Telemetry...';

    const formData = new FormData();
    formData.append('preset_id', currentPreset);
    formData.append('n_features', paramNFeatures ? paramNFeatures.value : '5000');
    formData.append('ratio_threshold', paramRatio ? paramRatio.value : '0.75');
    formData.append('reprojection_threshold', paramRansac ? paramRansac.value : '5.0');
    formData.append('clip_limit', paramClahe ? paramClahe.value : '2.0');

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
      if (metricRmse) metricRmse.textContent = m.rmse_pixels.toFixed(4);
      if (metricInlierRatio) metricInlierRatio.textContent = m.inlier_ratio.toFixed(2);
      if (metricInliers) metricInliers.textContent = m.inliers.toLocaleString();
      if (metricMatches) metricMatches.textContent = m.good_matches.toLocaleString();
      if (metricKeypoints) metricKeypoints.textContent = (m.source_keypoints + m.reference_keypoints).toLocaleString();
      if (metricSrcKp) metricSrcKp.textContent = m.source_keypoints.toLocaleString();
      if (metricRefKp) metricRefKp.textContent = m.reference_keypoints.toLocaleString();
      if (metricLatency) metricLatency.textContent = m.latency_ms;

      // Format Homography Matrix
      if (m.homography && matrixDisplay) {
        const matrixStr = m.homography
          .map((row) => '[ ' + row.map((v) => v.toFixed(6).padStart(12, ' ')).join(', ') + ' ]')
          .join('\n');
        matrixDisplay.textContent = matrixStr;
      }

      // Update Visualizer Images
      const imgs = data.images;
      if (imgRefBottom) imgRefBottom.src = imgs.reference;
      if (imgRegTop) imgRegTop.src = imgs.registered;
      if (imgMatches) imgMatches.src = imgs.matches;
      if (imgCheckerboard) imgCheckerboard.src = imgs.checkerboard;
      if (imgDifference) imgDifference.src = imgs.difference;
      if (imgBlendRef) imgBlendRef.src = imgs.reference;
      if (imgBlendReg) imgBlendReg.src = imgs.registered;

      if (imgRefBottom) {
        imgRefBottom.onload = syncSplitWidth;
      }
      syncSplitWidth();

      playSuccessPing();
      showToast(`🛰️ Telemetry Locked! RMSE: ${m.rmse_pixels.toFixed(4)} px (${m.inlier_ratio}% RANSAC inliers)`);
    } catch (err) {
      console.error(err);
      showToast(`❌ Telemetry Error: ${err.message}`);
    } finally {
      if (runBtn) runBtn.disabled = false;
      if (btnSpinner) btnSpinner.style.display = 'none';
      if (btnIcon) btnIcon.style.display = 'inline-block';
      if (btnText) btnText.textContent = 'Execute Registration Pipeline';
      if (window.lucide) {
        window.lucide.createIcons();
      }
    }
  }

  if (runBtn) {
    runBtn.addEventListener('click', executeRegistration);
  }

  // -------------------------------------------------------------
  // Download Output Handler
  // -------------------------------------------------------------
  if (downloadRegBtn) {
    downloadRegBtn.addEventListener('click', () => {
      playChirp(1050, 'triangle', 0.08);
      if (!lastRegistrationResult || !lastRegistrationResult.images.registered) {
        showToast('⚠️ Please run correspondence pipeline first.');
        return;
      }
      const link = document.createElement('a');
      link.href = lastRegistrationResult.images.registered;
      link.download = `chandrayaan2_registered_optical_${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      showToast('💾 High-resolution registered telemetry frame downloaded.');
    });
  }

  // Initial Run on Load
  executeRegistration();
});
