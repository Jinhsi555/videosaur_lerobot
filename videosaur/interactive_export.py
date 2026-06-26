# ruff: noqa: E501
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from videosaur import utils
from videosaur.visualizations import color_map


DEFAULT_FEATURE_KEY = "encoder.vit_block_keys12"
DEFAULT_PREDICTION_KEY = "decoder.reconstruction"
DEFAULT_FIXED_PROBABILITY_MAX = 0.4
DEFAULT_FIXED_DIFFERENCE_MAX = 0.25
MANIFEST_FILENAME = "manifest.json"


VIEWER_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VideoSAUR Interactive Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f1;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #68717c;
      --line: #d7dcd3;
      --teal: #167c80;
      --coral: #d65f42;
      --gold: #d29b2d;
      --shadow: 0 10px 30px rgba(31, 43, 56, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    .app {
      min-height: 100vh;
      padding: 18px;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 14px;
    }
    .topbar, .panel, .timeline {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.25;
      font-weight: 720;
    }
    .meta {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
    }
    button, select, input[type="range"] { font: inherit; }
    button, select {
      border: 1px solid var(--line);
      background: #fbfcf8;
      color: var(--ink);
      border-radius: 7px;
      min-height: 34px;
      padding: 6px 10px;
    }
    button {
      cursor: pointer;
      min-width: 38px;
    }
    button:hover, select:hover { border-color: #9aa796; }
    button.active {
      background: var(--teal);
      border-color: var(--teal);
      color: white;
    }
    .segmented {
      display: inline-grid;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fbfcf8;
    }
    .mode-segmented { grid-template-columns: repeat(4, auto); }
    .scale-segmented { grid-template-columns: repeat(2, auto); }
    .segmented button {
      border: 0;
      border-radius: 0;
      border-right: 1px solid var(--line);
      min-width: 116px;
    }
    .scale-segmented button { min-width: 92px; }
    .segmented button:last-child { border-right: 0; }
    .segmented button:disabled {
      cursor: not-allowed;
      opacity: 0.46;
    }
    .content {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 14px;
      min-height: 0;
    }
    .stage {
      display: grid;
      grid-template-columns: repeat(2, minmax(220px, 1fr));
      gap: 14px;
      align-items: start;
    }
    .panel {
      padding: 12px;
      min-width: 0;
    }
    .panel-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }
    .panel-title {
      font-size: 13px;
      font-weight: 730;
      text-transform: uppercase;
      color: #31404d;
    }
    .panel-note {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .frame-box {
      position: relative;
      width: 100%;
      background: #111820;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #1d2a35;
      aspect-ratio: 1 / 1;
    }
    .frame-box img, .grid-canvas, .heatmap-atlas {
      position: absolute;
      inset: 0;
    }
    .frame-box img {
      width: 100%;
      height: 100%;
      object-fit: fill;
      display: block;
    }
    .grid-canvas {
      width: 100%;
      height: 100%;
      display: block;
      pointer-events: none;
    }
    .heatmap-atlas {
      width: var(--tile-w, 28px);
      height: var(--tile-h, 28px);
      opacity: 0.74;
      image-rendering: pixelated;
      transform-origin: top left;
      pointer-events: none;
      display: none;
    }
    .side {
      display: grid;
      gap: 14px;
      align-content: start;
      min-width: 0;
    }
    .opacity-row {
      display: grid;
      grid-template-columns: auto minmax(90px, 1fr) 44px;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      margin-top: 10px;
    }
    .legend {
      display: grid;
      gap: 7px;
      max-height: 260px;
      overflow: auto;
    }
    .legend-item {
      display: grid;
      grid-template-columns: 14px 1fr auto;
      gap: 8px;
      align-items: center;
      min-height: 26px;
      font-size: 12px;
      color: #34414c;
      border-radius: 6px;
      padding: 4px 5px;
      cursor: pointer;
    }
    .legend-item.active {
      background: #eef7f5;
      outline: 1px solid rgba(22, 124, 128, 0.35);
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      border: 1px solid rgba(0,0,0,0.12);
    }
    .timeline {
      padding: 12px 14px;
      display: grid;
      grid-template-columns: auto minmax(180px, 1fr) auto;
      gap: 12px;
      align-items: center;
    }
    .timeline label {
      color: var(--muted);
      font-size: 12px;
    }
    .mask-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .mask-tile {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px;
      background: #fbfcf8;
      cursor: pointer;
    }
    .mask-tile.active {
      outline: 2px solid var(--teal);
      outline-offset: 1px;
    }
    .mask-tile img {
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      background: #111820;
      border-radius: 5px;
      object-fit: cover;
    }
    .mask-label {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-top: 6px;
      color: #34414c;
      font-size: 12px;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      min-height: 16px;
      margin-top: 8px;
    }
    @media (max-width: 980px) {
      .topbar, .content, .stage, .timeline {
        grid-template-columns: 1fr;
      }
      .controls {
        justify-content: flex-start;
      }
      .side {
        grid-template-columns: minmax(0, 1fr);
      }
      .segmented {
        width: 100%;
      }
      .mode-segmented { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .scale-segmented { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .segmented button {
        min-width: 0;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <header class="topbar">
      <div>
        <h1 id="title">VideoSAUR Viewer</h1>
        <div class="meta" id="meta">Loading</div>
      </div>
      <div class="controls">
        <div class="segmented mode-segmented" aria-label="heatmap mode">
          <button data-mode="feature" title="Raw feature affinity">Raw Affinity</button>
          <button data-mode="target" class="active" title="Feature target used by transition loss">Feature Target</button>
          <button data-mode="prediction" title="Decoder prediction">Decoder Prediction</button>
          <button data-mode="difference" title="Prediction minus feature target">Difference</button>
        </div>
        <div class="segmented scale-segmented" aria-label="heatmap color scale">
          <button data-scale="relative" class="active" title="Per-patch min-max color scale">Relative</button>
          <button data-scale="fixed" title="Fixed probability color scale">Fixed</button>
        </div>
        <button id="prev" title="Previous frame">Prev</button>
        <button id="play" title="Play or pause">Play</button>
        <button id="next" title="Next frame">Next</button>
        <select id="speed" title="Playback speed">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
    </header>

    <section class="content">
      <div>
        <div class="stage">
          <section class="panel">
            <div class="panel-header">
              <div class="panel-title">Source Frame</div>
              <div class="panel-note" id="source-note">frame 0</div>
            </div>
            <div class="frame-box" id="source-box">
              <img id="source-img" alt="">
              <canvas id="source-grid" class="grid-canvas"></canvas>
            </div>
            <div class="status" id="patch-status"></div>
          </section>
          <section class="panel">
            <div class="panel-header">
              <div class="panel-title">Target Heatmap</div>
              <div class="panel-note" id="target-note">frame 1</div>
            </div>
            <div class="frame-box" id="target-box">
              <img id="target-img" alt="">
              <div id="heatmap-overlay" class="heatmap-atlas"></div>
              <canvas id="target-grid" class="grid-canvas"></canvas>
            </div>
            <div class="status" id="heat-status"></div>
          </section>
        </div>
        <section class="panel" style="margin-top:14px;">
          <div class="panel-header">
            <div class="panel-title">Slot Masks</div>
            <div class="panel-note" id="mask-note">frame 0</div>
          </div>
          <div class="mask-grid" id="mask-grid"></div>
        </section>
      </div>
      <aside class="side">
        <section class="panel">
          <div class="panel-header">
            <div class="panel-title">Slot Overlay</div>
            <div class="panel-note" id="overlay-note">all slots</div>
          </div>
          <div class="frame-box" id="overlay-box">
            <img id="overlay-base-img" alt="">
            <img id="slot-overlay-img" alt="">
          </div>
          <div class="opacity-row">
            <label for="opacity">Opacity</label>
            <input id="opacity" type="range" min="0" max="1" value="0.7" step="0.05">
            <span id="opacity-value">0.70</span>
          </div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <div class="panel-title">Slot Coverage</div>
            <div class="panel-note" id="coverage-note">frame 0</div>
          </div>
          <div class="legend" id="legend"></div>
        </section>
      </aside>
    </section>

    <footer class="timeline">
      <label for="frame">Frame</label>
      <input id="frame" type="range" min="0" max="0" value="0">
      <div id="frame-readout">0 / 0</div>
    </footer>
  </main>

  <script>
    const state = {
      manifest: null,
      frame: 0,
      mode: 'target',
      scale: 'relative',
      lockedPatch: null,
      hoverPatch: 0,
      selectedSlot: null,
      playing: false,
      timer: null,
    };

    const els = {
      title: document.getElementById('title'),
      meta: document.getElementById('meta'),
      sourceBox: document.getElementById('source-box'),
      targetBox: document.getElementById('target-box'),
      overlayBox: document.getElementById('overlay-box'),
      sourceImg: document.getElementById('source-img'),
      targetImg: document.getElementById('target-img'),
      overlayBaseImg: document.getElementById('overlay-base-img'),
      slotOverlayImg: document.getElementById('slot-overlay-img'),
      sourceGrid: document.getElementById('source-grid'),
      targetGrid: document.getElementById('target-grid'),
      heatmapOverlay: document.getElementById('heatmap-overlay'),
      sourceNote: document.getElementById('source-note'),
      targetNote: document.getElementById('target-note'),
      patchStatus: document.getElementById('patch-status'),
      heatStatus: document.getElementById('heat-status'),
      frameInput: document.getElementById('frame'),
      frameReadout: document.getElementById('frame-readout'),
      prev: document.getElementById('prev'),
      play: document.getElementById('play'),
      next: document.getElementById('next'),
      speed: document.getElementById('speed'),
      opacity: document.getElementById('opacity'),
      opacityValue: document.getElementById('opacity-value'),
      maskGrid: document.getElementById('mask-grid'),
      legend: document.getElementById('legend'),
      maskNote: document.getElementById('mask-note'),
      overlayNote: document.getElementById('overlay-note'),
      coverageNote: document.getElementById('coverage-note'),
    };

    function sourceFrame() {
      const maxPredictionFrame = Math.max(0, state.manifest.prediction_frames - 1);
      return Math.min(state.frame, maxPredictionFrame);
    }

    function targetFrame() {
      const offset = state.manifest.prediction_target_offset || 1;
      return Math.min(sourceFrame() + offset, state.manifest.n_frames - 1);
    }

    function activePatch() {
      return state.lockedPatch !== null ? state.lockedPatch : state.hoverPatch;
    }

    function patchCoords(patch) {
      const [gridH, gridW] = state.manifest.patch_grid;
      return [Math.floor(patch / gridW), patch % gridW, gridH, gridW];
    }

    function patchFromEvent(event) {
      const rect = els.sourceBox.getBoundingClientRect();
      const [gridH, gridW] = state.manifest.patch_grid;
      const x = Math.max(0, Math.min(rect.width - 1, event.clientX - rect.left));
      const y = Math.max(0, Math.min(rect.height - 1, event.clientY - rect.top));
      const patchX = Math.min(gridW - 1, Math.floor(x / rect.width * gridW));
      const patchY = Math.min(gridH - 1, Math.floor(y / rect.height * gridH));
      return patchY * gridW + patchX;
    }

    function setAspectRatios() {
      const [height, width] = state.manifest.image_size;
      const ratio = `${width} / ${height}`;
      for (const box of [els.sourceBox, els.targetBox, els.overlayBox]) {
        box.style.aspectRatio = ratio;
      }
    }

    function sizeGridCanvas(canvas) {
      const [height, width] = state.manifest.image_size;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    }

    function drawGrid(canvas, patch) {
      sizeGridCanvas(canvas);
      const ctx = canvas.getContext('2d');
      const [height, width] = state.manifest.image_size;
      const [gridH, gridW] = state.manifest.patch_grid;
      const cellW = width / gridW;
      const cellH = height / gridH;
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.strokeStyle = 'rgba(255,255,255,0.24)';
      ctx.lineWidth = 0.5;
      for (let x = 1; x < gridW; x++) {
        ctx.beginPath();
        ctx.moveTo(x * cellW, 0);
        ctx.lineTo(x * cellW, height);
        ctx.stroke();
      }
      for (let y = 1; y < gridH; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * cellH);
        ctx.lineTo(width, y * cellH);
        ctx.stroke();
      }
      if (patch !== null) {
        const [py, px] = patchCoords(patch);
        ctx.strokeStyle = '#d29b2d';
        ctx.lineWidth = 2;
        ctx.strokeRect(px * cellW + 1, py * cellH + 1, cellW - 2, cellH - 2);
      }
      ctx.restore();
    }

    function updateFrameImages() {
      const src = sourceFrame();
      const tgt = targetFrame();
      els.sourceImg.src = state.manifest.frame_paths[src];
      els.targetImg.src = state.manifest.frame_paths[tgt];
      els.overlayBaseImg.src = state.manifest.frame_paths[state.frame];
      els.sourceNote.textContent = `frame ${src}`;
      els.targetNote.textContent = `frame ${tgt}`;
    }

    function activeHeatmapPaths() {
      const fixedPaths = state.manifest.heatmap_atlas_fixed_paths || {};
      if (state.scale === 'fixed' && fixedPaths[state.mode]) {
        return fixedPaths[state.mode];
      }
      return state.manifest.heatmap_atlas_paths[state.mode];
    }

    function updateHeatmap() {
      const patch = activePatch();
      if (patch === null || state.manifest.prediction_frames <= 0) {
        els.heatmapOverlay.style.display = 'none';
        els.patchStatus.textContent = '';
        els.heatStatus.textContent = '';
        drawGrid(els.sourceGrid, null);
        drawGrid(els.targetGrid, null);
        return;
      }

      const [py, px, gridH, gridW] = patchCoords(patch);
      const paths = activeHeatmapPaths();
      const atlasPath = paths[sourceFrame()];
      els.heatmapOverlay.style.display = 'block';
      els.heatmapOverlay.style.setProperty('--tile-w', `${gridW}px`);
      els.heatmapOverlay.style.setProperty('--tile-h', `${gridH}px`);
      els.heatmapOverlay.style.backgroundImage = `url("${atlasPath}")`;
      els.heatmapOverlay.style.backgroundPosition = `${-px * gridW}px ${-py * gridH}px`;
      els.heatmapOverlay.style.transform = `scale(${els.targetBox.clientWidth / gridW}, ${els.targetBox.clientHeight / gridH})`;

      drawGrid(els.sourceGrid, patch);
      drawGrid(els.targetGrid, null);
      els.patchStatus.textContent = `patch ${patch} (${px}, ${py})`;
      const labels = {
        feature: 'raw feature affinity',
        target: 'feature target from transition loss',
        prediction: 'decoder softmax',
        difference: 'prediction minus feature target',
      };
      const scaleLabels = {
        relative: 'relative color',
        fixed: 'fixed scale',
      };
      const label = labels[state.mode] || state.mode;
      const scale = scaleLabels[state.scale] || state.scale;
      els.heatStatus.textContent = `${label}, ${scale}, target frame ${targetFrame()}`;
    }

    function updateSlotOverlay() {
      const frame = state.frame;
      const opacity = Number(els.opacity.value);
      const paths = state.manifest.slot_overlay_paths;
      els.slotOverlayImg.src = state.selectedSlot === null
        ? paths.all[frame]
        : paths.slots[frame][state.selectedSlot];
      els.slotOverlayImg.style.opacity = opacity;
      els.overlayNote.textContent = state.selectedSlot === null ? 'all slots' : `slot ${state.selectedSlot}`;
      els.opacityValue.textContent = opacity.toFixed(2);
    }

    function drawLegend() {
      const frame = state.frame;
      els.legend.innerHTML = '';
      for (let slot = 0; slot < state.manifest.n_slots; slot++) {
        const row = document.createElement('button');
        row.className = 'legend-item' + (state.selectedSlot === slot ? ' active' : '');
        row.title = `Slot ${slot}`;
        row.addEventListener('click', () => {
          state.selectedSlot = state.selectedSlot === slot ? null : slot;
          render(false);
        });
        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = state.manifest.colors_hex[slot];
        const label = document.createElement('span');
        label.textContent = `Slot ${slot}`;
        const value = document.createElement('span');
        const coverage = state.manifest.slot_coverage[frame][slot] * 100;
        value.textContent = `${coverage.toFixed(1)}%`;
        row.append(swatch, label, value);
        els.legend.appendChild(row);
      }
      els.coverageNote.textContent = `frame ${frame}`;
    }

    function drawMaskGrid() {
      const frame = state.frame;
      els.maskGrid.innerHTML = '';
      for (let slot = 0; slot < state.manifest.n_slots; slot++) {
        const tile = document.createElement('button');
        tile.className = 'mask-tile' + (state.selectedSlot === slot ? ' active' : '');
        tile.title = `Slot ${slot}`;
        tile.addEventListener('click', () => {
          state.selectedSlot = state.selectedSlot === slot ? null : slot;
          render(false);
        });
        const img = document.createElement('img');
        img.src = state.manifest.mask_soft_paths[frame][slot];
        img.alt = `slot ${slot} mask`;
        img.style.borderTop = `4px solid ${state.manifest.colors_hex[slot]}`;
        const label = document.createElement('div');
        label.className = 'mask-label';
        const coverage = state.manifest.slot_coverage[frame][slot] * 100;
        label.innerHTML = `<span>Slot ${slot}</span><span>${coverage.toFixed(1)}%</span>`;
        tile.append(img, label);
        els.maskGrid.appendChild(tile);
      }
      els.maskNote.textContent = `frame ${frame}`;
    }

    function render(reloadFrameImages = true) {
      els.frameInput.value = String(state.frame);
      els.frameReadout.textContent = `${state.frame} / ${state.manifest.n_frames - 1}`;
      if (reloadFrameImages) updateFrameImages();
      updateHeatmap();
      updateSlotOverlay();
      drawLegend();
      drawMaskGrid();
    }

    function setFrame(frame) {
      state.frame = Math.max(0, Math.min(state.manifest.n_frames - 1, frame));
      render(true);
    }

    function setMode(mode) {
      state.mode = mode;
      for (const button of document.querySelectorAll('[data-mode]')) {
        button.classList.toggle('active', button.dataset.mode === mode);
      }
      updateHeatmap();
    }

    function setScale(scale) {
      const fixedPaths = state.manifest.heatmap_atlas_fixed_paths || {};
      if (scale === 'fixed' && !Object.keys(fixedPaths).length) return;
      state.scale = scale;
      for (const button of document.querySelectorAll('[data-scale]')) {
        button.classList.toggle('active', button.dataset.scale === scale);
      }
      updateHeatmap();
    }

    function togglePlay() {
      state.playing = !state.playing;
      els.play.textContent = state.playing ? 'Pause' : 'Play';
      if (state.timer) clearInterval(state.timer);
      if (!state.playing) return;
      const fps = Math.max(1, state.manifest.fps || 12) * Number(els.speed.value);
      state.timer = setInterval(() => {
        const next = state.frame + 1 >= state.manifest.n_frames ? 0 : state.frame + 1;
        setFrame(next);
      }, Math.max(30, 1000 / fps));
    }

    function bindEvents() {
      for (const button of document.querySelectorAll('[data-mode]')) {
        button.addEventListener('click', () => setMode(button.dataset.mode));
      }
      for (const button of document.querySelectorAll('[data-scale]')) {
        button.addEventListener('click', () => setScale(button.dataset.scale));
      }
      els.prev.addEventListener('click', () => setFrame(state.frame - 1));
      els.next.addEventListener('click', () => setFrame(state.frame + 1));
      els.play.addEventListener('click', togglePlay);
      els.speed.addEventListener('change', () => {
        if (state.playing) {
          togglePlay();
          togglePlay();
        }
      });
      els.frameInput.addEventListener('input', (event) => setFrame(Number(event.target.value)));
      els.opacity.addEventListener('input', updateSlotOverlay);
      els.sourceBox.addEventListener('mousemove', (event) => {
        if (state.lockedPatch !== null) return;
        const patch = patchFromEvent(event);
        if (patch !== state.hoverPatch) {
          state.hoverPatch = patch;
          updateHeatmap();
        }
      });
      els.sourceBox.addEventListener('mouseleave', () => {
        if (state.lockedPatch !== null) return;
        state.hoverPatch = null;
        updateHeatmap();
      });
      els.sourceBox.addEventListener('click', (event) => {
        const patch = patchFromEvent(event);
        state.lockedPatch = state.lockedPatch === patch ? null : patch;
        state.hoverPatch = patch;
        updateHeatmap();
      });
      window.addEventListener('resize', updateHeatmap);
      document.addEventListener('keydown', (event) => {
        if (event.key === 'ArrowLeft') setFrame(state.frame - 1);
        if (event.key === 'ArrowRight') setFrame(state.frame + 1);
        if (event.key === ' ') {
          event.preventDefault();
          togglePlay();
        }
      });
    }

    async function init() {
      const response = await fetch('manifest.json');
      state.manifest = await response.json();
      setAspectRatios();
      els.title.textContent = state.manifest.video_name || 'VideoSAUR Viewer';
      els.meta.textContent = `${state.manifest.n_frames} frames, ${state.manifest.n_slots} slots, ${state.manifest.patch_grid[1]} x ${state.manifest.patch_grid[0]} patches`;
      els.frameInput.max = String(state.manifest.n_frames - 1);
      const hasFixedScale = !!state.manifest.heatmap_atlas_fixed_paths;
      for (const button of document.querySelectorAll('[data-scale="fixed"]')) {
        button.disabled = !hasFixedScale;
        if (!hasFixedScale) button.title = 'Regenerate this viewer to enable fixed-scale atlases';
      }
      bindEvents();
      render(true);
    }

    init().catch((error) => {
      els.meta.textContent = String(error);
      console.error(error);
    });
  </script>
</body>
</html>
"""


def interactive_enabled(config) -> bool:
    output = config.get("output", {})
    interactive = output.get("interactive", {}) if output is not None else {}
    return bool(interactive.get("enabled", False))


def resolve_viewer_dir(config) -> Path:
    output = config.get("output", {})
    interactive = output.get("interactive", {}) if output is not None else {}
    viewer_dir = interactive.get("viewer_dir")
    if viewer_dir:
        return Path(str(viewer_dir))

    save_path = output.get("save_path") if output is not None else None
    if save_path:
        save_path = Path(str(save_path))
        return save_path.with_name(f"{save_path.stem}-viewer")

    input_path = Path(str(config.input.path))
    return Path("inference_output") / f"{input_path.stem}-viewer"


def export_interactive_viewer(config, model_config, inputs, outputs, aux_outputs) -> Path:
    if str(config.input.type) != "video":
        raise ValueError("Interactive viewer export currently supports video inputs only.")

    output = config.get("output", {})
    interactive = output.get("interactive", {}) if output is not None else {}
    debug_arrays = bool(interactive.get("debug_arrays", False))
    viewer_dir = resolve_viewer_dir(config)
    _prepare_viewer_dir(viewer_dir, debug_arrays=debug_arrays)

    video = inputs["video_visualization"].detach().cpu()
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(
            "Expected inputs['video_visualization'] with shape [1, 3, frames, height, width]."
        )

    _, _, n_frames, height, width = video.shape
    frames_dir = viewer_dir / "frames"
    soft_dir = viewer_dir / "masks_soft"
    hard_dir = viewer_dir / "masks_hard"
    heatmaps_dir = viewer_dir / "heatmaps"
    fixed_heatmaps_dir = viewer_dir / "heatmaps_fixed"
    overlays_dir = viewer_dir / "slot_overlays"

    frame_paths = _write_frames(video, frames_dir)

    feature_key = str(interactive.get("feature_key", DEFAULT_FEATURE_KEY))
    features = _read_tensor_path(outputs, feature_key)
    if features is None and feature_key != "encoder.backbone_features":
        feature_key = "encoder.backbone_features"
        features = _read_tensor_path(outputs, feature_key)
    if features is None:
        raise ValueError("Could not find encoder patch features for interactive viewer export.")
    features = _validate_video_features(features, n_frames)
    n_patches = features.shape[1]
    patch_grid = infer_patch_grid(n_patches, (height, width))

    prediction_key = str(interactive.get("prediction_key", DEFAULT_PREDICTION_KEY))
    prediction = _read_tensor_path(outputs, prediction_key)
    if prediction is None:
        raise ValueError(f"Could not find decoder prediction tensor at `{prediction_key}`.")
    prediction = _validate_video_prediction(prediction, n_frames, n_patches)
    prediction_dims = resolve_prediction_dims(
        interactive,
        model_config,
        prediction.shape[-1],
        n_patches,
    )
    similarity_config = resolve_similarity_config(model_config)
    time_shift = int(similarity_config.get("time_shift", 1))
    remove_last_n_frames = resolve_prediction_remove_last_n_frames(model_config, time_shift)
    prediction_frames = min(
        max(0, n_frames - remove_last_n_frames),
        max(0, n_frames - time_shift),
    )
    prediction_logits = prediction[:prediction_frames, :, prediction_dims[0] : prediction_dims[1]]

    masks_soft = _decoder_masks_for_export(outputs, aux_outputs, (height, width), hard=False)
    masks_hard = _decoder_masks_for_export(outputs, aux_outputs, (height, width), hard=True)
    if masks_soft.shape[:2] != masks_hard.shape[:2]:
        raise ValueError("Soft and hard decoder mask dimensions differ.")
    if masks_soft.shape[0] != n_frames:
        raise ValueError(
            f"Decoder masks have {masks_soft.shape[0]} frames, but video has {n_frames}."
        )
    n_slots = masks_soft.shape[1]
    mask_soft_paths = _write_masks(masks_soft, soft_dir)
    mask_hard_paths = _write_masks(masks_hard, hard_dir)

    colors_rgb = [[int(channel) for channel in color] for color in color_map(n_slots)]
    colors_hex = ["#{:02x}{:02x}{:02x}".format(*color) for color in colors_rgb]
    slot_coverage = masks_hard.reshape(n_frames, n_slots, -1).mean(axis=-1).tolist()
    slot_overlay_paths = _write_slot_overlays(masks_hard, colors_rgb, overlays_dir)
    fixed_scale_config = resolve_fixed_heatmap_scale(interactive)
    heatmap_atlas_paths, heatmap_atlas_fixed_paths = _write_heatmap_atlases(
        features,
        prediction_logits,
        patch_grid,
        similarity_config,
        heatmaps_dir,
        fixed_heatmaps_dir,
        fixed_scale_config,
    )

    arrays_manifest = {}
    if debug_arrays:
        arrays_dir = viewer_dir / "arrays"
        np.save(arrays_dir / "features.npy", features.astype(np.float16, copy=False))
        np.save(arrays_dir / "prediction_logits.npy", prediction_logits.astype(np.float16, copy=False))
        arrays_manifest = {
            "features": "arrays/features.npy",
            "prediction_logits": "arrays/prediction_logits.npy",
        }

    manifest = {
        "schema_version": 2,
        "viewer_type": "videosaur_static_atlas",
        "video_name": Path(str(config.input.path)).name,
        "input_path": str(config.input.path),
        "output_path": str(output.get("save_path", "")),
        "fps": float(config.get("fps", 0) or 0),
        "n_frames": int(n_frames),
        "n_slots": int(n_slots),
        "n_patches": int(n_patches),
        "image_size": [int(height), int(width)],
        "patch_grid": [int(patch_grid[0]), int(patch_grid[1])],
        "atlas_tile_size": [int(patch_grid[0]), int(patch_grid[1])],
        "atlas_grid": [int(patch_grid[0]), int(patch_grid[1])],
        "frame_paths": frame_paths,
        "mask_soft_paths": mask_soft_paths,
        "mask_hard_paths": mask_hard_paths,
        "slot_overlay_paths": slot_overlay_paths,
        "heatmap_atlas_paths": heatmap_atlas_paths,
        "heatmap_atlas_fixed_paths": heatmap_atlas_fixed_paths,
        "heatmap_fixed_scale": fixed_scale_config,
        "slot_coverage": slot_coverage,
        "colors_rgb": colors_rgb,
        "colors_hex": colors_hex,
        "feature_key": feature_key,
        "prediction_key": prediction_key,
        "prediction_dims": [int(prediction_dims[0]), int(prediction_dims[1])],
        "prediction_frames": int(prediction_logits.shape[0]),
        "prediction_remove_last_n_frames": int(remove_last_n_frames),
        "prediction_target_offset": int(time_shift),
        "similarity": similarity_config,
    }
    if arrays_manifest:
        manifest["arrays"] = arrays_manifest

    with (viewer_dir / MANIFEST_FILENAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    (viewer_dir / "index.html").write_text(VIEWER_INDEX_HTML, encoding="utf-8")
    return viewer_dir


def compute_frame_heatmaps(
    features: np.ndarray,
    prediction_logits: np.ndarray,
    frame: int,
    similarity_config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    if frame < 0 or frame >= prediction_logits.shape[0]:
        raise ValueError(f"Frame {frame} is outside prediction range.")
    time_shift = int(similarity_config.get("time_shift", 1))
    if frame + time_shift >= features.shape[0]:
        raise ValueError(f"Frame {frame} has no target frame at offset {time_shift}.")
    feature_affinity = _feature_similarity_matrix(
        features[frame],
        features[frame + time_shift],
        normalize=bool(similarity_config.get("normalize", True)),
    )
    target_similarity = feature_affinity
    feature_target = _feature_target_transform(target_similarity, similarity_config)
    prediction = _softmax_rows(prediction_logits[frame])
    return {
        "feature": feature_affinity.astype(np.float32, copy=False),
        "target": feature_target.astype(np.float32, copy=False),
        "prediction": prediction.astype(np.float32, copy=False),
        "difference": (prediction - feature_target).astype(np.float32, copy=False),
    }


def resolve_prediction_dims(
    interactive_config,
    model_config,
    prediction_dim: int,
    n_patches: int,
) -> Tuple[int, int]:
    configured = interactive_config.get("prediction_dims") if interactive_config is not None else None
    if configured is not None:
        return _validate_prediction_dims(configured, prediction_dim, n_patches)

    loss_config = _find_time_similarity_loss(model_config)
    if loss_config is not None and loss_config.get("pred_dims") is not None:
        return _validate_prediction_dims(loss_config.get("pred_dims"), prediction_dim, n_patches)

    if prediction_dim == n_patches:
        return (0, n_patches)
    if prediction_dim >= 2 * n_patches:
        return (prediction_dim - n_patches, prediction_dim)
    if prediction_dim >= n_patches:
        return (0, n_patches)

    raise ValueError(
        f"Prediction dimension {prediction_dim} is smaller than the patch count {n_patches}."
    )


def resolve_similarity_config(model_config) -> Dict[str, Any]:
    loss_config = _find_time_similarity_loss(model_config)
    transform = loss_config.get("target_transform") if loss_config is not None else None
    if transform is None:
        return {
            "time_shift": 1,
            "normalize": True,
            "temperature": 1.0,
            "threshold": None,
            "mask_diagonal": False,
            "softmax": True,
        }

    threshold = transform.get("threshold")
    return {
        "time_shift": int(transform.get("time_shift", 1)),
        "normalize": bool(transform.get("normalize", True)),
        "temperature": float(transform.get("temperature", 1.0)),
        "threshold": None if threshold is None else float(threshold),
        "mask_diagonal": bool(transform.get("mask_diagonal", False)),
        "softmax": bool(transform.get("softmax", True)),
    }


def resolve_prediction_remove_last_n_frames(model_config, time_shift: int = 1) -> int:
    loss_config = _find_time_similarity_loss(model_config)
    if loss_config is None:
        return int(time_shift)
    return int(loss_config.get("remove_last_n_frames", time_shift))


def resolve_fixed_heatmap_scale(interactive_config) -> Dict[str, float]:
    probability_max = float(
        interactive_config.get("fixed_probability_max", DEFAULT_FIXED_PROBABILITY_MAX)
        if interactive_config is not None
        else DEFAULT_FIXED_PROBABILITY_MAX
    )
    difference_abs_max = float(
        interactive_config.get("fixed_difference_max", DEFAULT_FIXED_DIFFERENCE_MAX)
        if interactive_config is not None
        else DEFAULT_FIXED_DIFFERENCE_MAX
    )
    if probability_max <= 0:
        raise ValueError("output.interactive.fixed_probability_max must be > 0.")
    if difference_abs_max <= 0:
        raise ValueError("output.interactive.fixed_difference_max must be > 0.")
    return {
        "probability_max": probability_max,
        "difference_abs_max": difference_abs_max,
    }


def infer_patch_grid(n_patches: int, image_size: Tuple[int, int]) -> Tuple[int, int]:
    height, width = image_size
    ratio = width / height
    grid_h = int(math.sqrt(n_patches / ratio))
    grid_w = int(math.sqrt(n_patches * ratio))
    if grid_h * grid_w != n_patches:
        square = int(math.sqrt(n_patches))
        if square * square == n_patches:
            return (square, square)
        raise ValueError(f"Can not infer patch grid for {n_patches} patches.")
    return (grid_h, grid_w)


def _prepare_viewer_dir(viewer_dir: Path, debug_arrays: bool):
    viewer_dir.mkdir(parents=True, exist_ok=True)
    for child in (
        "frames",
        "arrays",
        "masks_soft",
        "masks_hard",
        "heatmaps",
        "heatmaps_fixed",
        "slot_overlays",
    ):
        path = viewer_dir / child
        if path.exists():
            shutil.rmtree(path)

    for child in ("frames", "masks_soft", "masks_hard", "heatmaps", "heatmaps_fixed", "slot_overlays"):
        (viewer_dir / child).mkdir(parents=True, exist_ok=True)
    if debug_arrays:
        (viewer_dir / "arrays").mkdir(parents=True, exist_ok=True)


def _write_frames(video: torch.Tensor, frames_dir: Path):
    paths = []
    for frame_idx in range(video.shape[2]):
        frame = _frame_to_uint8(video[0, :, frame_idx])
        rel_path = f"frames/frame_{frame_idx:06d}.jpg"
        imageio.imwrite(frames_dir / f"frame_{frame_idx:06d}.jpg", frame, quality=92)
        paths.append(rel_path)
    return paths


def _write_masks(masks: np.ndarray, masks_dir: Path):
    all_paths = []
    for frame_idx in range(masks.shape[0]):
        frame_paths = []
        for slot_idx in range(masks.shape[1]):
            rel_path = f"{masks_dir.name}/frame_{frame_idx:06d}_slot_{slot_idx:02d}.png"
            mask = np.clip(masks[frame_idx, slot_idx] * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(masks_dir / f"frame_{frame_idx:06d}_slot_{slot_idx:02d}.png", mask)
            frame_paths.append(rel_path)
        all_paths.append(frame_paths)
    return all_paths


def _write_slot_overlays(masks_hard: np.ndarray, colors_rgb, overlays_dir: Path):
    all_dir = overlays_dir / "all"
    slots_dir = overlays_dir / "slots"
    all_dir.mkdir(parents=True, exist_ok=True)
    slots_dir.mkdir(parents=True, exist_ok=True)
    all_paths = []
    slot_paths = []
    colors = np.asarray(colors_rgb, dtype=np.uint8)

    for frame_idx in range(masks_hard.shape[0]):
        frame_slots = []
        all_overlay = np.zeros((*masks_hard.shape[-2:], 4), dtype=np.uint8)
        for slot_idx in range(masks_hard.shape[1]):
            mask = masks_hard[frame_idx, slot_idx] > 0.5
            overlay = np.zeros((*masks_hard.shape[-2:], 4), dtype=np.uint8)
            overlay[mask, :3] = colors[slot_idx]
            overlay[mask, 3] = 255
            all_overlay[mask, :3] = colors[slot_idx]
            all_overlay[mask, 3] = 255
            rel_path = f"slot_overlays/slots/frame_{frame_idx:06d}_slot_{slot_idx:02d}.png"
            imageio.imwrite(overlays_dir / "slots" / f"frame_{frame_idx:06d}_slot_{slot_idx:02d}.png", overlay)
            frame_slots.append(rel_path)
        rel_all = f"slot_overlays/all/frame_{frame_idx:06d}.png"
        imageio.imwrite(all_dir / f"frame_{frame_idx:06d}.png", all_overlay)
        all_paths.append(rel_all)
        slot_paths.append(frame_slots)

    return {"all": all_paths, "slots": slot_paths}


def _write_heatmap_atlases(
    features: np.ndarray,
    prediction_logits: np.ndarray,
    patch_grid: Tuple[int, int],
    similarity_config: Dict[str, Any],
    heatmaps_dir: Path,
    fixed_heatmaps_dir: Path,
    fixed_scale_config: Dict[str, float],
):
    paths = {"feature": [], "target": [], "prediction": [], "difference": []}
    fixed_paths = {"target": [], "prediction": [], "difference": []}
    for kind in paths:
        (heatmaps_dir / kind).mkdir(parents=True, exist_ok=True)
    for kind in fixed_paths:
        (fixed_heatmaps_dir / kind).mkdir(parents=True, exist_ok=True)

    for frame_idx in range(prediction_logits.shape[0]):
        heatmaps = compute_frame_heatmaps(features, prediction_logits, frame_idx, similarity_config)
        for kind, values in heatmaps.items():
            atlas = _heatmap_values_to_atlas(values, patch_grid, kind)
            rel_path = f"heatmaps/{kind}/frame_{frame_idx:06d}.png"
            imageio.imwrite(heatmaps_dir / kind / f"frame_{frame_idx:06d}.png", atlas)
            paths[kind].append(rel_path)
            if kind in fixed_paths:
                fixed_atlas = _heatmap_values_to_fixed_atlas(
                    values,
                    patch_grid,
                    kind,
                    fixed_scale_config,
                )
                fixed_rel_path = f"heatmaps_fixed/{kind}/frame_{frame_idx:06d}.png"
                imageio.imwrite(
                    fixed_heatmaps_dir / kind / f"frame_{frame_idx:06d}.png",
                    fixed_atlas,
                )
                fixed_paths[kind].append(fixed_rel_path)
    return paths, fixed_paths


def _heatmap_values_to_atlas(
    values: np.ndarray,
    patch_grid: Tuple[int, int],
    kind: str,
) -> np.ndarray:
    if kind == "feature":
        t = np.clip((values + 1.0) / 2.0, 0.0, 1.0)
        colors = _interpolate_stops(t, ([49, 95, 157], [245, 246, 241], [214, 95, 66]))
    elif kind == "difference":
        max_abs = np.max(np.abs(values), axis=1, keepdims=True)
        t = np.zeros_like(values, dtype=np.float32)
        np.divide(values, max_abs, out=t, where=max_abs > 1e-12)
        t = np.clip((t + 1.0) / 2.0, 0.0, 1.0)
        colors = _interpolate_stops(t, ([49, 95, 157], [245, 246, 241], [214, 95, 66]))
    else:
        row_min = np.min(values, axis=1, keepdims=True)
        row_max = np.max(values, axis=1, keepdims=True)
        denom = row_max - row_min
        t = np.full_like(values, 0.5, dtype=np.float32)
        np.divide(values - row_min, denom, out=t, where=denom > 1e-12)
        colors = _interpolate_stops(
            np.clip(t, 0.0, 1.0),
            ([19, 60, 85], [54, 145, 143], [241, 221, 149], [214, 95, 66]),
        )

    return _colors_to_atlas(colors, patch_grid)


def _heatmap_values_to_fixed_atlas(
    values: np.ndarray,
    patch_grid: Tuple[int, int],
    kind: str,
    fixed_scale_config: Dict[str, float],
) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    if kind == "difference":
        max_abs = float(fixed_scale_config["difference_abs_max"])
        t = np.clip((values / max_abs + 1.0) / 2.0, 0.0, 1.0)
        colors = _interpolate_stops(t, ([49, 95, 157], [245, 246, 241], [214, 95, 66]))
    else:
        probability_max = float(fixed_scale_config["probability_max"])
        t = np.clip(values / probability_max, 0.0, 1.0)
        colors = _interpolate_stops(
            t,
            ([19, 60, 85], [54, 145, 143], [241, 221, 149], [214, 95, 66]),
        )

    return _colors_to_atlas(colors, patch_grid)


def _colors_to_atlas(colors: np.ndarray, patch_grid: Tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = patch_grid
    atlas = np.zeros((grid_h * grid_h, grid_w * grid_w, 3), dtype=np.uint8)
    for source_patch in range(grid_h * grid_w):
        source_y, source_x = divmod(source_patch, grid_w)
        tile = colors[source_patch].reshape(grid_h, grid_w, 3)
        y0 = source_y * grid_h
        x0 = source_x * grid_w
        atlas[y0 : y0 + grid_h, x0 : x0 + grid_w] = tile
    return atlas


def _interpolate_stops(t: np.ndarray, stops) -> np.ndarray:
    stop_array = np.asarray(stops, dtype=np.float32)
    scaled = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    idx = np.floor(scaled).astype(np.int64)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = scaled - idx
    a = stop_array[idx]
    b = stop_array[idx + 1]
    return np.rint(a + (b - a) * frac[..., None]).astype(np.uint8)


def _frame_to_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu()
    if torch.is_floating_point(frame):
        if frame.max() <= 1.0:
            frame = frame * 255
        frame = frame.clamp(0, 255)
    return frame.to(torch.uint8).permute(1, 2, 0).numpy()


def _read_tensor_path(root, path: str) -> Optional[torch.Tensor]:
    value = utils.read_path(root, path=path, error=False)
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise ValueError(f"Expected tensor at `{path}`, got {type(value)}.")
    return value.detach().cpu()


def _validate_video_features(features: torch.Tensor, n_frames: int) -> np.ndarray:
    if features.ndim != 4 or features.shape[0] != 1 or features.shape[1] != n_frames:
        raise ValueError(
            "Expected features with shape [1, frames, patches, dim], "
            f"got {tuple(features.shape)}."
        )
    return features[0].float().numpy()


def _validate_video_prediction(
    prediction: torch.Tensor,
    n_frames: int,
    n_patches: int,
) -> np.ndarray:
    if prediction.ndim != 4 or prediction.shape[0] != 1 or prediction.shape[1] != n_frames:
        raise ValueError(
            "Expected prediction with shape [1, frames, patches, dim], "
            f"got {tuple(prediction.shape)}."
        )
    if prediction.shape[2] != n_patches:
        raise ValueError(
            f"Prediction patch dimension {prediction.shape[2]} does not match {n_patches}."
        )
    return prediction[0].float().numpy()


def _decoder_masks_for_export(outputs, aux_outputs, image_size: Tuple[int, int], hard: bool):
    key_order = (
        ("decoder_masks_vis_hard", "decoder_masks_hard") if hard else ("decoder_masks",)
    )
    masks = None
    for key in key_order:
        masks = aux_outputs.get(key) if aux_outputs is not None else None
        if masks is not None:
            break

    if masks is None and not hard:
        masks = outputs["decoder"].get("masks")
    if masks is None and hard:
        soft_masks = _decoder_masks_for_export(outputs, aux_outputs, image_size, hard=False)
        argmax = np.argmax(soft_masks, axis=1)
        return np.eye(soft_masks.shape[1], dtype=np.float32)[argmax].transpose(0, 3, 1, 2)

    masks = masks.detach().cpu().float()
    if masks.ndim == 4:
        b, t, n_slots, n_patches = masks.shape
        grid = infer_patch_grid(n_patches, image_size)
        masks = masks.reshape(b, t, n_slots, grid[0], grid[1])
    elif masks.ndim != 5:
        raise ValueError(f"Expected decoder masks with 4 or 5 dims, got {tuple(masks.shape)}.")

    if masks.shape[0] != 1:
        raise ValueError("Interactive viewer export supports batch size 1 only.")
    if masks.shape[-2:] != image_size:
        b, t, n_slots, _, _ = masks.shape
        mode = "nearest" if hard else "bilinear"
        masks_flat = masks.reshape(b * t, n_slots, masks.shape[-2], masks.shape[-1])
        if mode == "nearest":
            masks_flat = F.interpolate(masks_flat, size=image_size, mode=mode)
        else:
            masks_flat = F.interpolate(masks_flat, size=image_size, mode=mode, align_corners=False)
        masks = masks_flat.reshape(b, t, n_slots, image_size[0], image_size[1])

    masks = masks[0].numpy()
    if hard:
        masks = (masks > 0.5).astype(np.float32)
    else:
        masks = np.clip(masks, 0.0, 1.0).astype(np.float32)
    return masks


def _feature_similarity_matrix(
    source: np.ndarray,
    target: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    source = source.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)
    if normalize:
        source_norm = np.linalg.norm(source, axis=-1, keepdims=True)
        target_norm = np.linalg.norm(target, axis=-1, keepdims=True)
        source = source / np.maximum(source_norm, 1e-12)
        target = target / np.maximum(target_norm, 1e-12)
    return source @ target.T


def _feature_target_transform(values: np.ndarray, similarity_config) -> np.ndarray:
    temperature = float(similarity_config.get("temperature", 1.0) or 1.0)
    softmax = bool(similarity_config.get("softmax", True))
    padding_value = -np.inf if softmax else -1.0 / temperature
    transformed = values.astype(np.float32, copy=True)

    threshold = similarity_config.get("threshold")
    if threshold is not None:
        transformed[transformed < float(threshold)] = padding_value

    transformed /= temperature

    if similarity_config.get("mask_diagonal", False):
        np.fill_diagonal(transformed, padding_value)

    if softmax:
        return _softmax_rows(transformed)
    return transformed


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    safe = np.where(finite, values, -np.inf)
    row_max = np.max(safe, axis=-1, keepdims=True)
    empty_rows = ~np.isfinite(row_max)
    row_max = np.where(empty_rows, 0.0, row_max)
    exp = np.where(finite, np.exp(safe - row_max), 0.0).astype(np.float32)
    if empty_rows.any():
        exp = exp.copy()
        exp[np.broadcast_to(empty_rows, exp.shape)] = 1.0
    total = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.maximum(total, 1e-12)


def _validate_prediction_dims(configured, prediction_dim: int, n_patches: int) -> Tuple[int, int]:
    if len(configured) != 2:
        raise ValueError(f"Expected prediction_dims pair, got {configured}.")
    start, end = int(configured[0]), int(configured[1])
    if start < 0 or end > prediction_dim or start >= end:
        raise ValueError(
            f"Invalid prediction_dims [{start}, {end}] for prediction dimension {prediction_dim}."
        )
    if end - start != n_patches:
        raise ValueError(
            f"prediction_dims [{start}, {end}] span {end - start} values, "
            f"but expected {n_patches} patches."
        )
    return (start, end)


def _find_time_similarity_loss(model_config):
    model = model_config.get("model") if model_config is not None else None
    losses = model.get("losses") if model is not None else None
    if losses is None:
        return None

    if losses.get("loss_timesim") is not None:
        return losses.get("loss_timesim")

    for _, loss_config in losses.items():
        transform = loss_config.get("target_transform")
        if transform is None:
            continue
        if str(transform.get("name", "")).endswith("FeatureTimeSimilarity"):
            return loss_config

    return None
