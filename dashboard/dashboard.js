/**
 * dashboard.js — RoadSense Adaptive Suspension Dashboard
 *
 * Connects to the FastAPI backend via:
 *   • WebSocket  ws://localhost:8000/ws/live   → real-time prediction data
 *   • MJPEG      /api/stream/video             → live camera feed
 *   • REST       /api/status, /api/session/stats, /api/config/*
 */

(() => {
  'use strict';

  // ── Config ──────────────────────────────────────────────────────────────
  const API_BASE = 'http://localhost:8000';
  const WS_URL   = 'ws://localhost:8000/ws/live';

  // ── DOM Refs ────────────────────────────────────────────────────────────
  const $ = (id) => document.getElementById(id);

  const dom = {
    // Top bar
    pulseIndicator:  $('pulse-indicator'),
    chipEngine:      $('chip-engine'),
    chipCamera:      $('chip-camera'),
    chipModel:       $('chip-model'),
    metricFps:       $('metric-fps'),
    metricLatency:   $('metric-latency'),

    // Suspension
    suspensionMode:   $('suspension-mode'),
    suspensionAction: $('suspension-action'),
    paramStiffness:   $('param-stiffness'),
    paramDamping:     $('param-damping'),
    gaugeStiffness:   $('gauge-stiffness'),

    // Camera
    cameraFeed:    $('camera-feed'),
    feedStatus:    $('feed-status'),
    overlayClass:  $('overlay-class'),
    overlayConf:   $('overlay-conf'),

    // Probabilities
    probModeLabel: $('prob-mode-label'),

    // Model
    modelType: $('model-type'),
    modelPath: $('model-path'),

    // Controls
    gradcamCheckbox:  $('gradcam-checkbox'),
    smoothedCheckbox: $('smoothed-checkbox'),

    // History
    historyCanvas: $('history-canvas'),

    // Session
    sessionDuration:    $('session-duration'),
    sessionPredictions: $('session-predictions'),
  };


  // ── State ───────────────────────────────────────────────────────────────
  let ws = null;
  let useSmoothed = true;
  let predictionCount = 0;
  let sessionStart = Date.now();
  let historyBuffer = [];     // last 60 s of class labels
  const MAX_HISTORY = 300;    // ~60 s at ~5 updates/s

  // Class → color mapping (matching design.md)
  const CLASS_COLORS = {
    smooth:  '#CCFF00',
    gravel:  '#00E0FF',
    pothole: '#ffb4ab',
    wet:     '#FFAB00',
  };


  // ── Camera Feed (MJPEG) ─────────────────────────────────────────────────
  function startCameraFeed() {
    const img = dom.cameraFeed;
    // MJPEG stream — the browser natively handles multipart/x-mixed-replace
    img.src = `${API_BASE}/api/stream/video`;

    img.onload = () => {
      dom.feedStatus.textContent = 'LIVE';
      dom.feedStatus.classList.add('connected');
    };

    img.onerror = () => {
      dom.feedStatus.textContent = 'OFFLINE';
      dom.feedStatus.classList.remove('connected');
      // Retry after 3 s
      setTimeout(() => { img.src = `${API_BASE}/api/stream/video?t=${Date.now()}`; }, 3000);
    };
  }


  // ── WebSocket ───────────────────────────────────────────────────────────
  function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('[WS] Connected');
      setChipState(dom.chipEngine, 'active');
      // Request data-only (no embedded frame — we use MJPEG)
      ws.send(JSON.stringify({ include_frame: false }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handlePrediction(data);
      } catch (err) {
        console.warn('[WS] Parse error:', err);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected — reconnecting in 2 s');
      setChipState(dom.chipEngine, 'error');
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }


  // ── Handle Prediction Data ──────────────────────────────────────────────
  function handlePrediction(data) {
    predictionCount++;

    const source    = useSmoothed ? data.smoothed : data.raw;
    const probs     = source?.probabilities || {};
    const predicted = source?.class || '—';
    const confidence = source?.confidence || 0;
    const suspension = data.suspension || {};
    const metrics    = data.metrics || {};

    // ── Probability Bars ──────────────────────────────────────────────
    updateBars(probs, predicted);

    // ── Camera Overlay ────────────────────────────────────────────────
    dom.overlayClass.textContent = predicted.toUpperCase();
    dom.overlayConf.textContent  = `${(confidence * 100).toFixed(1)}%`;
    dom.overlayClass.style.color = CLASS_COLORS[predicted] || '#e5e2e3';

    // ── Suspension Panel ──────────────────────────────────────────────
    const mode = suspension.mode || '—';
    dom.suspensionMode.textContent  = mode;
    dom.suspensionMode.setAttribute('data-mode', mode);
    dom.suspensionAction.textContent = suspension.recommended_action || '';

    const stiffness = suspension.stiffness || 0;
    const damping   = suspension.damping || 0;
    dom.paramStiffness.innerHTML = `${stiffness.toLocaleString()} <span class="param-unit">N/m</span>`;
    dom.paramDamping.innerHTML   = `${damping.toLocaleString()} <span class="param-unit">Ns/m</span>`;

    // Update arc gauge (0–15000 range for stiffness)
    updateGauge(stiffness, mode);

    // ── Top-bar Metrics ───────────────────────────────────────────────
    dom.metricFps.textContent = (metrics.fps || 0).toFixed(1);
    dom.metricLatency.innerHTML = `${(metrics.latency_ms || 0).toFixed(0)}<span class="metric-inline__unit">ms</span>`;

    // ── Model chip ────────────────────────────────────────────────────
    if (metrics.model_type && metrics.model_type !== 'none') {
      dom.modelType.textContent = metrics.model_type.toUpperCase();
      setChipState(dom.chipModel, 'active');
    }

    // ── History Buffer ────────────────────────────────────────────────
    historyBuffer.push({
      t: Date.now(),
      cls: predicted,
      conf: confidence,
    });
    if (historyBuffer.length > MAX_HISTORY) historyBuffer.shift();
  }


  // ── Update Probability Bars ─────────────────────────────────────────────
  function updateBars(probs, activeClass) {
    for (const cls of ['smooth', 'gravel', 'pothole', 'wet']) {
      const pct = ((probs[cls] || 0) * 100).toFixed(1);
      const bar = $(`bar-${cls}`);
      const val = $(`val-${cls}`);
      if (bar) bar.style.width = `${pct}%`;
      if (val) val.textContent = `${pct}%`;

      // Mark active row
      const row = bar?.closest('.prob-row');
      if (row) {
        row.classList.toggle('active', cls === activeClass);
      }
    }
  }


  // ── Arc Gauge ───────────────────────────────────────────────────────────
  function updateGauge(stiffness, mode) {
    const maxStiffness = 15000;
    const fraction = Math.min(stiffness / maxStiffness, 1);
    const arcLen   = 251.3;  // matches SVG path length
    const offset   = arcLen * (1 - fraction);

    dom.gaugeStiffness.style.strokeDashoffset = offset;

    // Color by mode
    const colorMap = {
      Soft:     '#CCFF00',
      Medium:   '#00E0FF',
      Firm:     '#ffb4ab',
      Adaptive: '#FFAB00',
    };
    dom.gaugeStiffness.style.stroke = colorMap[mode] || '#CCFF00';
  }


  // ── History Timeline (Canvas) ───────────────────────────────────────────
  function drawHistory() {
    const canvas = dom.historyCanvas;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    // High-DPI
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;

    ctx.clearRect(0, 0, W, H);

    if (historyBuffer.length < 2) {
      ctx.fillStyle = '#c4c9ac';
      ctx.font = '12px "Space Grotesk", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for data…', W / 2, H / 2);
      return;
    }

    const classIndices = { smooth: 0, gravel: 1, pothole: 2, wet: 3 };
    const bandH = H / 4;

    // Draw horizontal class bands
    for (const [cls, idx] of Object.entries(classIndices)) {
      const y = idx * bandH;
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      if (idx % 2 === 0) ctx.fillRect(0, y, W, bandH);

      ctx.fillStyle = '#444933';
      ctx.font = '9px "Space Grotesk", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(cls.toUpperCase(), 4, y + 12);
    }

    // Plot dots
    const n = historyBuffer.length;
    const step = W / Math.max(n - 1, 1);

    for (let i = 0; i < n; i++) {
      const entry = historyBuffer[i];
      const cls = entry.cls;
      const idx = classIndices[cls] ?? 0;
      const x = i * step;
      const y = idx * bandH + bandH / 2;
      const r = Math.max(2, 3 * entry.conf);

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = CLASS_COLORS[cls] || '#e5e2e3';
      ctx.globalAlpha = 0.5 + 0.5 * entry.conf;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }


  // ── Chip State Helper ───────────────────────────────────────────────────
  function setChipState(chip, state) {
    chip.className = 'chip';
    if (state === 'active')  chip.classList.add('chip--active');
    if (state === 'warning') chip.classList.add('chip--warning');
    if (state === 'error')   chip.classList.add('chip--error');
  }


  // ── Poll Status ─────────────────────────────────────────────────────────
  async function pollStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (!res.ok) return;
      const s = await res.json();

      // Camera chip
      setChipState(dom.chipCamera, s.camera_active ? 'active' : 'error');

      // Model chip
      if (s.model_loaded) {
        setChipState(dom.chipModel, 'active');
        dom.modelType.textContent = (s.model_type || 'unknown').toUpperCase();
        dom.modelPath.textContent = s.model_path || '—';
      } else {
        setChipState(dom.chipModel, 'warning');
        dom.modelType.textContent = 'NONE';
        dom.modelPath.textContent = 'No model loaded';
      }

      // Session duration
      const dur = s.session_duration || 0;
      const mm = String(Math.floor(dur / 60)).padStart(2, '0');
      const ss = String(Math.floor(dur % 60)).padStart(2, '0');
      dom.sessionDuration.textContent = `${mm}:${ss}`;
      dom.sessionPredictions.textContent = predictionCount.toLocaleString();
    } catch {
      // Server offline
      setChipState(dom.chipEngine, 'error');
      setChipState(dom.chipCamera, 'error');
    }
  }


  // ── Controls ────────────────────────────────────────────────────────────
  dom.gradcamCheckbox.addEventListener('change', async () => {
    const enabled = dom.gradcamCheckbox.checked;
    try {
      await fetch(`${API_BASE}/api/config/gradcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
    } catch (err) {
      console.warn('GradCAM toggle failed:', err);
    }

    // Also tell the WebSocket to include GradCAM in the stream
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ gradcam: enabled }));
    }
  });

  dom.smoothedCheckbox.addEventListener('change', () => {
    useSmoothed = dom.smoothedCheckbox.checked;
    dom.probModeLabel.textContent = useSmoothed ? 'SMOOTHED' : 'RAW';
  });


  // ── Render Loop ─────────────────────────────────────────────────────────
  function renderLoop() {
    drawHistory();
    requestAnimationFrame(renderLoop);
  }


  // ── Init ────────────────────────────────────────────────────────────────
  function init() {
    startCameraFeed();
    connectWebSocket();
    pollStatus();
    setInterval(pollStatus, 3000);

    // Kick off canvas render
    requestAnimationFrame(renderLoop);

    // Initial toggle states
    dom.probModeLabel.textContent = 'SMOOTHED';
  }

  // Go
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
