"""Serve the existing ZMQ camera stream as a small browser-based MJPEG UI.

The HTTP server binds to loopback by default. Access it from another computer
through an SSH local forward instead of exposing the viewer on the network::

    ssh -N -L 8080:127.0.0.1:8080 unitree@ROBOT_HOST

Then open http://127.0.0.1:8080 in a local browser.
"""

from dataclasses import dataclass
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import re
import signal
import subprocess
import threading
import time

import cv2
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
import numpy as np
import tyro
import zmq

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorClient
from gear_sonic.end_effectors.protocol import (
    DEFAULT_HAND_CONTROL_PORT,
    DEFAULT_HAND_STATE_PORT,
    HAND_CONTROL_SCHEMA,
    HAND_CONTROL_TOPIC,
    HAND_STATE_TOPIC,
    decode_state,
    encode,
)
from gear_sonic.utils.data_collection.hub_config import (
    DEFAULT_HF_NAMESPACE,
    DEFAULT_TASK_PROMPT,
    encode_dataset_config,
)
from gear_sonic.utils.teleop.pico_body_diagnostics import DEFAULT_PICO_BODY_PORT, PicoBodySubscriber

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SONIC Teleoperation</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #11151a; color: #edf2f7; }
    header { display: flex; align-items: center; gap: 12px; padding: 14px 18px;
             background: #1b222b; border-bottom: 1px solid #303a46; }
    h1 { margin: 0; font-size: 17px; font-weight: 600; }
    #camera-status { margin-left: auto; font-size: 13px; color: #f6c85f; }
    main { min-height: calc(100vh - 58px); display: flex; flex-direction: column;
           align-items: center; justify-content: center; gap: 14px;
           padding: 18px; box-sizing: border-box; }
    img { display: block; max-width: 100%; max-height: calc(100vh - 175px);
          border-radius: 6px; background: #090b0e; box-shadow: 0 8px 30px #0008; }
    .control-card { width: min(1100px, 100%); display: flex; align-items: center;
                gap: 12px; padding: 12px 14px; box-sizing: border-box;
                background: #1b222b; border: 1px solid #303a46; border-radius: 8px; }
    #record-state { min-width: 96px; padding: 6px 10px; text-align: center;
                    font-weight: 700; border-radius: 999px; background: #303a46; }
    #record-state.recording { background: #b4232f; color: white; animation: pulse 1.2s infinite; }
    #record-state.saving { background: #9a6700; color: white; }
    #hand-state { min-width: 96px; padding: 6px 10px; text-align: center;
                  font-weight: 700; border-radius: 999px; background: #303a46; }
    #hand-state.connected { background: #287a4d; color: white; }
    #hand-state.recovering { background: #9a6700; color: white; }
    @keyframes pulse { 50% { opacity: .65; } }
    .recorder-info { min-width: 0; flex: 1; }
    #record-message { font-size: 14px; }
    #record-detail { margin-top: 3px; color: #9eabb8; font-size: 12px;
                     white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .rate-card { display: block; }
    .rate-title { margin-bottom: 8px; font-size: 14px; font-weight: 700; }
    .rate-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    .rate-table th, .rate-table td { padding: 5px 8px; text-align: right;
                                     border-top: 1px solid #303a46; font-size: 12px; }
    .rate-table th:first-child, .rate-table td:first-child { text-align: left; }
    .rate-table th { color: #9eabb8; font-weight: 600; }
    .rate-ok { color: #72d69c; }
    .rate-alert { color: #ef7777; font-weight: 700; }
    .rate-idle { color: #788594; }
    .rate-note { margin-top: 7px; color: #788594; font-size: 11px; }
    .dataset-card { align-items: flex-start; }
    .dataset-fields { display: grid; grid-template-columns: minmax(240px, 1fr) auto;
                      gap: 9px 12px; width: 100%; }
    .dataset-fields label { color: #9eabb8; font-size: 12px; }
    .dataset-fields .wide { grid-column: 1 / -1; }
    .repo-line { display: flex; align-items: center; gap: 6px; margin-top: 4px; }
    .repo-prefix { color: #9eabb8; white-space: nowrap; }
    input[type="text"], textarea { width: 100%; box-sizing: border-box; border-radius: 5px;
      border: 1px solid #465362; background: #10151b; color: #edf2f7; padding: 8px; }
    textarea { min-height: 68px; resize: vertical; font-family: inherit; }
    .dataset-actions { display: flex; align-items: center; justify-content: space-between;
                       gap: 12px; grid-column: 1 / -1; }
    #dataset-message { color: #9eabb8; font-size: 12px; overflow-wrap: anywhere; }
    #dataset-state { min-width: 96px; padding: 6px 10px; text-align: center;
                     font-weight: 700; border-radius: 999px; background: #303a46; }
    #dataset-state.ready { background: #287a4d; }
    #dataset-state.uploading { background: #356ca5; }
    #dataset-state.error { background: #b4232f; }
    button { border: 0; border-radius: 6px; padding: 10px 15px; color: white;
             font-weight: 650; cursor: pointer; background: #287a4d; }
    button.stop { background: #b4232f; }
    button.danger { background: #b4232f; }
    button.discard { background: #59636f; }
    button:disabled { cursor: not-allowed; opacity: .4; }
    #pico-panel summary { cursor: pointer; font-size: 14px; font-weight: 650; }
    #pico-state { margin-left: 12px; font-size: 12px; }
    .pico-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
    .pico-toolbar select { background: #11151a; color: #edf2f7; padding: 7px; border: 1px solid #465362; border-radius: 5px; }
    .pico-content { display: grid; grid-template-columns: minmax(0, 1fr) minmax(240px, 1fr); gap: 14px; }
    #pico-canvas { width: 100%; background: #10151b; border-radius: 6px; }
    .pico-joints { max-height: 360px; overflow: auto; }
    #pico-detail { font-size: 12px; color: #9eabb8; overflow-wrap: anywhere; }
    @media (max-width: 700px) { .pico-content { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>SONIC · G1 Teleoperation</h1><span id="camera-status">connecting…</span></header>
  <main>
    <img id="stream" src="/stream.mjpg" alt="Waiting for camera stream">
    <details id="pico-panel" class="control-card rate-card">
      <summary>Raw PICO body · received <span id="pico-rate">— Hz</span> · <span id="pico-state" class="rate-idle">waiting…</span></summary>
      <div class="pico-toolbar">
        <label>View <select id="pico-projection"><option value="0,1">X / Y</option><option value="0,2">X / Z</option><option value="1,2">Y / Z</option></select></label>
        <button id="pico-fit" type="button">Fit view</button>
        <span id="pico-freshness"></span>
      </div>
      <div class="pico-content">
        <canvas id="pico-canvas" width="640" height="360" role="img" aria-label="Raw PICO body joint positions"></canvas>
        <div class="pico-joints"><table class="rate-table">
          <thead><tr><th>Joint</th><th>X (m)</th><th>Y (m)</th><th>Z (m)</th></tr></thead>
          <tbody id="pico-joints"></tbody>
        </table></div>
      </div>
      <div id="pico-detail"></div>
      <div class="rate-note">Positions received from PICO, before robot alignment or motion limiting. The last frame remains visible when stale. Standing still is normal; freshness is checked using the body timestamp.</div>
    </details>
    <section class="control-card dataset-card">
      <span id="dataset-state">SETUP</span>
      <div class="dataset-fields">
        <label>Hugging Face dataset repository
          <div class="repo-line">
            <span id="repo-prefix" class="repo-prefix">MicroAGI-Labs/</span>
            <input id="dataset-repo" type="text" list="dataset-repos" maxlength="96"
                   autocomplete="off" placeholder="episode-dataset">
            <datalist id="dataset-repos"></datalist>
          </div>
        </label>
        <label><input id="dataset-private" type="checkbox" checked> Private dataset</label>
        <label class="wide">Task prompt
          <textarea id="dataset-prompt" maxlength="1000">__DEFAULT_TASK_PROMPT__</textarea>
        </label>
        <div class="dataset-actions">
          <span id="dataset-message">Choose an existing empty repo or enter a new name.</span>
          <button id="dataset-configure" disabled>Create / Select</button>
        </div>
      </div>
    </section>
    <section class="control-card">
      <span id="record-state">CONNECTING</span>
      <div class="recorder-info">
        <div id="record-message">Waiting for recorder…</div>
        <div id="record-detail"></div>
      </div>
      <button id="record-toggle" disabled>Start Recording</button>
      <button id="record-discard" class="discard" disabled>Discard</button>
    </section>
    <section class="control-card rate-card">
      <div class="rate-title">Collector stream rates · rolling 2 seconds</div>
      <table class="rate-table">
        <thead><tr><th>Publisher → receiver</th><th>Publisher Hz</th><th>Receiver Hz</th></tr></thead>
        <tbody id="rate-body"><tr><td>Waiting for recorder…</td><td>—</td><td>—</td></tr></tbody>
      </table>
      <div class="rate-note">
        Publisher Hz is measured from source timestamps; receiver Hz is measured where
        the arrow ends. Wrist publisher rates estimate capture cadence; wrist receiver
        rates count fresh frames sampled by the collector. Inactive streams show 0 Hz.
      </div>
    </section>
    <section id="hand-controls" class="control-card" hidden>
      <span id="hand-state">CONNECTING</span>
      <div class="recorder-info">
        <div id="hand-message">Waiting for hand controller…</div>
        <div id="hand-detail"></div>
      </div>
      <button id="hand-reconnect" disabled>Reconnect Hands</button>
    </section>
    <section id="sonic-controls" class="control-card">
      <span id="sonic-state">SONIC</span>
      <div class="recorder-info">
        <div id="sonic-message">Policy process control</div>
        <div id="sonic-detail">Safe idle uses the smooth base-pose path. Disconnect stops SONIC.</div>
      </div>
      <button id="sonic-safe-idle">Return Safely to Idle</button>
      <button id="sonic-disconnect" class="danger">Disconnect SONIC</button>
    </section>
  </main>
  <script>
    const picoPanel = document.getElementById('pico-panel');
    const picoCanvas = document.getElementById('pico-canvas');
    const picoContext = picoCanvas.getContext('2d');
    const picoProjection = document.getElementById('pico-projection');
    const picoNames = ['Pelvis','Left hip','Right hip','Spine 1','Left knee','Right knee','Spine 2','Left ankle','Right ankle','Spine 3','Left foot','Right foot','Neck','Left collar','Right collar','Head','Left shoulder','Right shoulder','Left elbow','Right elbow','Left wrist','Right wrist','Left hand','Right hand'];
    const picoParents = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21];
    const picoRows = picoNames.map((name, i) => {
      const row = document.createElement('tr');
      for (const text of [`${i} · ${name}`, '—', '—', '—']) {
        const cell = document.createElement('td'); cell.textContent = text; row.appendChild(cell);
      }
      document.getElementById('pico-joints').appendChild(row); return row;
    });
    let picoData = null, picoBounds = null;
    function drawPico() {
      const ctx = picoContext, width = picoCanvas.width, height = picoCanvas.height;
      ctx.clearRect(0, 0, width, height);
      const poses = picoData && picoData.poses;
      if (!Array.isArray(poses) || poses.length !== 24 || !poses.every(p => Array.isArray(p) && p.length === 7 && p.every(Number.isFinite))) {
        ctx.fillStyle = '#9eabb8'; ctx.font = '16px system-ui'; ctx.fillText('Waiting for raw body positions…', 24, 40); return;
      }
      const [axisX, axisY] = picoProjection.value.split(',').map(Number);
      if (!picoBounds) {
        const xs = poses.map(p => p[axisX]), ys = poses.map(p => p[axisY]);
        const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
        picoBounds = {cx: (minX + maxX) / 2, cy: (minY + maxY) / 2,
          scale: Math.min((width - 90) / Math.max(maxX - minX, .4), (height - 70) / Math.max(maxY - minY, .4))};
      }
      const project = p => [width / 2 + (p[axisX] - picoBounds.cx) * picoBounds.scale,
                             height / 2 - (p[axisY] - picoBounds.cy) * picoBounds.scale];
      const points = poses.map(project);
      ctx.strokeStyle = picoData.body_live ? '#78bdf2' : '#697b8c'; ctx.lineWidth = 2;
      for (let i = 1; i < points.length; i++) {
        const parent = points[picoParents[i]]; ctx.beginPath(); ctx.moveTo(...parent); ctx.lineTo(...points[i]); ctx.stroke();
      }
      ctx.font = '11px system-ui';
      points.forEach(([x, y], i) => {
        ctx.fillStyle = [20,21].includes(i) ? '#f6c85f' : '#edf2f7';
        ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill(); ctx.fillText(String(i), x + 5, y - 5);
        for (let a = 0; a < 3; a++) picoRows[i].children[a + 1].textContent = poses[i][a].toFixed(4);
      });
      ctx.fillStyle = '#9eabb8'; ctx.fillText(`horizontal: ${'XYZ'[axisX]} · vertical: ${'XYZ'[axisY]} · metres`, 15, height - 12);
    }
    document.getElementById('pico-fit').addEventListener('click', () => {picoBounds = null; drawPico();});
    picoProjection.addEventListener('change', () => {picoBounds = null; drawPico();});
    picoPanel.addEventListener('toggle', drawPico);
    async function pollPico() {
      const state = document.getElementById('pico-state');
      try {
        const response = await fetch('/pico/body', {cache: 'no-store', signal: AbortSignal.timeout(3000)});
        if (!response.ok) throw new Error('PICO diagnostics unavailable');
        picoData = await response.json();
        const labels = {waiting: 'Waiting for input manager', disconnected: 'Input disconnected', unavailable: 'No body data',
          timestamp_unavailable: 'Body timestamp unavailable', live: 'Body streaming', stale: 'Body timestamp frozen'};
        state.textContent = labels[picoData.state] || 'Waiting';
        state.className = picoData.body_live ? 'rate-ok' : 'rate-alert';
        const rate = picoData.body_read_hz;
        document.getElementById('pico-rate').textContent = Number.isFinite(rate)
          ? `${rate.toFixed(1)} Hz` : '— Hz';
        const age = value => Number.isFinite(value) ? `${value.toFixed(1)}s` : '—';
        document.getElementById('pico-freshness').textContent = `Packets: ${picoData.packet_live ? 'live' : 'stale'} · body timestamp age: ${age(picoData.body_age_s)}`;
        document.getElementById('pico-detail').textContent = `Body timestamp: ${picoData.body_timestamp_ns || 'unavailable'} · Packet timestamp: ${picoData.packet_timestamp_ns || 'unavailable'}${picoData.error ? ' · ' + picoData.error : ''}`;
        if (picoPanel.open) drawPico();
      } catch (error) {
        state.textContent = 'Diagnostics unavailable'; state.className = 'rate-alert';
        document.getElementById('pico-rate').textContent = '— Hz';
        if (picoData) picoData.body_live = false;
        if (picoPanel.open) drawPico();
      } finally { setTimeout(pollPico, picoPanel.open ? 100 : 1000); }
    }
    pollPico();
    const cameraStatus = document.getElementById('camera-status');
    const recordState = document.getElementById('record-state');
    const recordMessage = document.getElementById('record-message');
    const recordDetail = document.getElementById('record-detail');
    const recordToggle = document.getElementById('record-toggle');
    const recordDiscard = document.getElementById('record-discard');
    const datasetState = document.getElementById('dataset-state');
    const datasetRepo = document.getElementById('dataset-repo');
    const datasetRepos = document.getElementById('dataset-repos');
    const datasetPrompt = document.getElementById('dataset-prompt');
    const datasetPrivate = document.getElementById('dataset-private');
    const datasetConfigure = document.getElementById('dataset-configure');
    const datasetMessage = document.getElementById('dataset-message');
    const repoPrefix = document.getElementById('repo-prefix');
    const rateBody = document.getElementById('rate-body');
    const handState = document.getElementById('hand-state');
    const handControls = document.getElementById('hand-controls');
    const handMessage = document.getElementById('hand-message');
    const handDetail = document.getElementById('hand-detail');
    const handReconnect = document.getElementById('hand-reconnect');
    const sonicState = document.getElementById('sonic-state');
    const sonicMessage = document.getElementById('sonic-message');
    const sonicSafeIdle = document.getElementById('sonic-safe-idle');
    const sonicDisconnect = document.getElementById('sonic-disconnect');
    let commandPending = false;
    let configurationPending = false;
    let datasetSetupError = null;
    let hydratedRepo = null;
    let lastRecorderStatus = null;
    let handCommandPending = false;
    const rateLabels = {
      camera: 'Camera server → collector',
      left_wrist: 'Left wrist camera → collector',
      right_wrist: 'Right wrist camera → collector',
      robot_state: 'Robot/C++ → collector',
      pico_pose: 'PICO manager (pose) → collector',
      planner: 'PICO manager (planner) → collector',
      manager_state: 'PICO manager (state) → collector',
      hand_intent: 'PICO manager (hand intent) → hand controller',
      hand_control: 'Hand controller (I/O loop) → state publisher',
      hand_state: 'Hand controller (state) → collector',
    };
    const collectorReceiverStreams = new Set([
      'camera', 'left_wrist', 'right_wrist', 'robot_state', 'pico_pose', 'planner',
      'manager_state', 'hand_state'
    ]);
    const minimumCollectorReceiverHz = 45;

    function formatHz(value) {
      return Number.isFinite(value) ? `${value.toFixed(2)} Hz` : '—';
    }

    function renderRates(rates) {
      rateBody.replaceChildren();
      for (const [name, label] of Object.entries(rateLabels)) {
        const rate = rates[name] || {};
        const row = document.createElement('tr');
        row.className = rate.active ? 'rate-ok' : 'rate-idle';
        const values = [label, formatHz(rate.sent_hz), formatHz(rate.received_hz)];
        for (const [index, text] of values.entries()) {
          const cell = document.createElement('td');
          cell.textContent = text;
          if (index === 2 && rate.active && collectorReceiverStreams.has(name)
              && Number.isFinite(rate.received_hz)
              && rate.received_hz < minimumCollectorReceiverHz) {
            cell.className = 'rate-alert';
          }
          row.appendChild(cell);
        }
        rateBody.appendChild(row);
      }
    }

    async function updateCameraStatus() {
      try {
        const response = await fetch('/healthz', {cache: 'no-store'});
        const health = await response.json();
        cameraStatus.textContent = health.streaming
          ? `${health.camera_count} camera${health.camera_count === 1 ? '' : 's'} · live`
          : 'waiting for camera…';
        cameraStatus.style.color = health.streaming ? '#72d69c' : '#f6c85f';
      } catch (_) {
        cameraStatus.textContent = 'viewer disconnected';
        cameraStatus.style.color = '#ef7777';
      }
    }

    async function updateRecorderStatus() {
      try {
        const response = await fetch('/recording/status', {cache: 'no-store'});
        const status = await response.json();
        lastRecorderStatus = status;
        if (!status.connected) {
          renderRates({});
          recordState.textContent = 'OFFLINE';
          recordState.className = '';
          recordMessage.textContent = 'Recorder unavailable';
          recordDetail.textContent = '';
          recordToggle.disabled = true;
          recordDiscard.disabled = true;
          updateDatasetStatus(status);
          return;
        }
        const state = status.recording ? 'RECORDING' : (status.saving ? 'SAVING' : 'IDLE');
        recordState.textContent = state;
        recordState.className = status.recording ? 'recording' : (status.saving ? 'saving' : '');
        recordMessage.textContent = status.message || state;
        renderRates(status.stream_rates || {});
        const sources = status.sources || {};
        const sourcesReady = sources.proprio && sources.camera && sources.hands;
        const modeReady = status.recording_mode_ready !== false;
        const ready = sourcesReady && modeReady;
        const hub = status.hub || {};
        const readiness = !sourcesReady
          ? 'source missing'
          : !modeReady
            ? `enter ${status.required_stream_mode_name || 'teleop'} with A+X`
            : 'sources and teleop ready';
        recordDetail.textContent =
          `episode ${status.episode_index} · ${status.frame_count} frames · ` +
          `${status.dataset_root} · ${readiness} · ` +
          (status.recording
            ? 'headset: release A+X or X+B to save, release Y+A to discard'
            : 'headset: release X+B to start; A+X twice toggles teleop');
        recordToggle.textContent = status.recording ? 'Stop & Save' : 'Start Recording';
        recordToggle.className = status.recording ? 'stop' : '';
        recordToggle.disabled = commandPending || status.saving ||
          (!status.recording && (!ready || (hub.required && !hub.ready)));
        recordDiscard.disabled = commandPending || !status.recording;
        updateDatasetStatus(status);
      } catch (_) {
        recordMessage.textContent = 'Recorder status request failed';
      }
    }

    function updateDatasetStatus(status) {
      const connected = Boolean(status.connected);
      const hub = status.hub || {};
      const finalizer = status.finalizer || {};
      const locked = Boolean(status.recording || status.saving || status.total_episodes > 0 ||
        finalizer.finalizing || finalizer.pending || hub.uploading || hub.pending);
      if (hub.ready && hub.repo_id && hydratedRepo !== hub.repo_id) {
        datasetRepo.value = hub.repo_id.split('/').slice(1).join('/');
        datasetPrompt.value = hub.prompt || datasetPrompt.value;
        datasetPrivate.checked = hub.private !== false;
        hydratedRepo = hub.repo_id;
      }
      if (!datasetRepo.value && status.dataset_root) {
        datasetRepo.value = status.dataset_root.split('/').filter(Boolean).pop() || '';
      }
      datasetRepo.disabled = locked;
      datasetPrompt.disabled = locked;
      datasetPrivate.disabled = locked;
      datasetConfigure.disabled = configurationPending || !connected || locked;

      if (hub.uploading) {
        datasetState.textContent = 'UPLOADING';
        datasetState.className = 'uploading';
        const queued = hub.pending > 1 ? ` · ${hub.pending} episodes queued` : '';
        datasetMessage.textContent =
          `Uploading to ${hub.repo_id}${queued}… you can keep recording.`;
      } else if (hub.retrying || hub.error) {
        datasetState.textContent = 'RETRYING';
        datasetState.className = 'error';
        datasetMessage.textContent =
          `Upload error (recording still allowed): ${hub.error || 'retrying automatically'}`;
      } else if (hub.ready) {
        datasetState.textContent = 'READY';
        datasetState.className = 'ready';
        const uploaded = Number.isInteger(hub.last_uploaded_episode)
          ? ` · episode ${hub.last_uploaded_episode} uploaded` : '';
        datasetMessage.textContent = `${hub.repo_id}${uploaded}`;
      } else {
        datasetState.textContent = datasetSetupError ? 'ERROR' : (connected ? 'SETUP' : 'OFFLINE');
        datasetState.className = datasetSetupError ? 'error' : '';
        datasetMessage.textContent = datasetSetupError || (connected
          ? 'Choose an existing empty repo or enter a new name.'
          : 'Recorder must be online before configuring the dataset.');
      }
    }

    async function loadDatasetRepos() {
      try {
        const response = await fetch('/datasets/repos', {cache: 'no-store'});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Could not list repositories');
        repoPrefix.textContent = `${payload.namespace}/`;
        datasetRepos.replaceChildren();
        for (const repoId of payload.repos || []) {
          const option = document.createElement('option');
          option.value = repoId.split('/').slice(1).join('/');
          datasetRepos.appendChild(option);
        }
      } catch (error) {
        datasetMessage.textContent = error.message;
      }
    }

    async function configureDataset() {
      configurationPending = true;
      datasetSetupError = null;
      datasetConfigure.disabled = true;
      datasetMessage.textContent = 'Creating/selecting repository…';
      try {
        const response = await fetch('/dataset/configure', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            repo: datasetRepo.value,
            prompt: datasetPrompt.value,
            private: datasetPrivate.checked,
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Dataset setup failed');
        datasetSetupError = null;
        datasetMessage.textContent = `${payload.repo_id} selected; waiting for recorder…`;
        setTimeout(updateRecorderStatus, 250);
      } catch (error) {
        datasetSetupError = error.message;
        datasetState.textContent = 'ERROR';
        datasetState.className = 'error';
        datasetMessage.textContent = error.message;
      } finally {
        configurationPending = false;
        if (lastRecorderStatus) updateDatasetStatus(lastRecorderStatus);
      }
    }

    async function sendRecorderCommand(path) {
      commandPending = true;
      recordToggle.disabled = true;
      recordDiscard.disabled = true;
      try {
        await fetch(path, {method: 'POST'});
      } finally {
        setTimeout(() => { commandPending = false; updateRecorderStatus(); }, 350);
      }
    }

    async function updateHandStatus() {
      try {
        const response = await fetch('/hands/status', {cache: 'no-store'});
        const status = await response.json();
        handControls.hidden = !status.enabled;
        if (!status.enabled) return;
        const sides = status.sides || {};
        const sideNames = Object.keys(sides);
        const connectedSides = sideNames.filter(side => sides[side].connected);
        const fullyConnected = status.connected && status.mode !== 'fault'
          && status.mode !== 'disconnected' && sideNames.length > 0
          && connectedSides.length === sideNames.length;
        handState.textContent = fullyConnected ? 'CONNECTED' : 'RECOVERING';
        handState.className = fullyConnected ? 'connected' : 'recovering';
        handMessage.textContent = fullyConnected
          ? `Hands ready · ${connectedSides.join(' + ')}`
          : (status.error || 'Hand worker is reconnecting…');
        const errors = sideNames
          .filter(side => sides[side].error)
          .map(side => `${side}: ${sides[side].error}`);
        handDetail.textContent = errors.length
          ? errors.join(' · ')
          : `mode: ${status.mode || 'offline'} · automatic recovery enabled`;
        handReconnect.disabled = handCommandPending;
      } catch (_) {
        handState.textContent = 'OFFLINE';
        handState.className = 'recovering';
        handMessage.textContent = 'Hand status request failed';
        handReconnect.disabled = false;
      }
    }

    async function reconnectHands() {
      handCommandPending = true;
      handReconnect.disabled = true;
      try {
        await fetch('/hands/reconnect', {method: 'POST'});
        handMessage.textContent = 'Clean hand-worker restart requested…';
      } finally {
        setTimeout(() => { handCommandPending = false; updateHandStatus(); }, 500);
      }
    }

    async function disconnectSonic() {
      const confirmed = window.confirm(
        'Disconnect SONIC from the robot? This stops the policy process and requires a relaunch.'
      );
      if (!confirmed) return;
      sonicDisconnect.disabled = true;
      const response = await fetch('/sonic/disconnect', {
        method: 'POST',
        headers: {'X-Sonic-Confirmation': 'disconnect'},
      });
      if (response.ok) {
        sonicState.textContent = 'STOPPING';
        sonicMessage.textContent = 'SONIC disconnect command sent';
      } else {
        sonicDisconnect.disabled = false;
        sonicMessage.textContent = 'Disconnect request failed';
      }
    }

    async function returnSonicToIdle() {
      sonicSafeIdle.disabled = true;
      try {
        const response = await fetch('/teleop/safe-idle', {method: 'POST'});
        if (!response.ok) throw new Error('request rejected');
        sonicState.textContent = 'RETURNING';
        sonicMessage.textContent = 'Smooth return through the base pose requested…';
        setTimeout(() => {
          sonicState.textContent = 'SONIC';
          sonicMessage.textContent = 'Policy process control';
          sonicSafeIdle.disabled = false;
        }, 4500);
      } catch (_) {
        sonicMessage.textContent = 'Safe-idle request failed';
        sonicSafeIdle.disabled = false;
      }
    }

    recordToggle.addEventListener('click', () => sendRecorderCommand('/recording/toggle'));
    recordDiscard.addEventListener('click', () => sendRecorderCommand('/recording/discard'));
    datasetConfigure.addEventListener('click', configureDataset);
    for (const field of [datasetRepo, datasetPrompt, datasetPrivate]) {
      field.addEventListener('input', () => { datasetSetupError = null; });
    }
    handReconnect.addEventListener('click', reconnectHands);
    sonicSafeIdle.addEventListener('click', returnSonicToIdle);
    sonicDisconnect.addEventListener('click', disconnectSonic);
    loadDatasetRepos(); updateCameraStatus(); updateRecorderStatus(); updateHandStatus();
    setInterval(updateCameraStatus, 1500);
    setInterval(updateRecorderStatus, 500);
    setInterval(updateHandStatus, 500);
  </script>
</body>
</html>
""".replace("__DEFAULT_TASK_PROMPT__", html.escape(DEFAULT_TASK_PROMPT)).encode("utf-8")


@dataclass
class CameraWebViewerConfig:
    """Configuration for the browser camera viewer."""

    camera_host: str = "localhost"
    """ZMQ camera publisher hostname."""

    camera_port: int = 5555
    """ZMQ camera publisher port."""

    http_host: str = "127.0.0.1"
    """HTTP bind address. Keep loopback when using an SSH forward."""

    http_port: int = 8080
    """HTTP port forwarded to the user's computer."""

    fps: int = 20
    """Maximum browser stream frame rate."""

    jpeg_quality: int = 80
    """JPEG quality used for the browser stream."""

    max_tile_width: int = 960
    """Maximum width of each camera tile."""

    recording_command_port: int = 5580
    """ZMQ PUB port used to send recorder commands."""

    recording_status_host: str = "localhost"
    """Host publishing authoritative recorder status."""

    recording_status_port: int = 5581
    """ZMQ SUB port used to receive recorder status."""

    hand_state_host: str = "localhost"
    """Host publishing external-hand controller state."""

    hand_state_port: int = DEFAULT_HAND_STATE_PORT
    """ZMQ SUB port used to receive external-hand state."""

    hand_control_port: int = DEFAULT_HAND_CONTROL_PORT
    """ZMQ PUB port used to request a clean hand-worker reconnect."""

    teleop_control_port: int = 5573
    """ZMQ PUB port used for safety commands to the teleop manager."""

    pico_body_host: str = "localhost"
    """Host publishing raw PICO body diagnostics."""

    pico_body_port: int = DEFAULT_PICO_BODY_PORT
    """Local diagnostics port shared with the PICO input manager."""

    enable_hand_controls: bool = False
    """Display and enable external-hand status and reconnect controls."""

    sonic_tmux_target: str = "sonic_data_collection:data_collection.0"
    """Tmux pane receiving Ctrl-C after an explicit UI disconnect confirmation."""

    hf_namespace: str = DEFAULT_HF_NAMESPACE
    """Hugging Face organization that owns selectable dataset repositories."""


class CameraFrameHub:
    """Receive camera messages once and fan the latest JPEG out to browsers."""

    def __init__(self, config: CameraWebViewerConfig):
        self.config = config
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._camera_count = 0
        self._last_frame_time = 0.0
        self._running = True
        self._client = SensorClient()
        self._client.start_client(config.camera_host, config.camera_port)
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._client.stop_client()
        with self._condition:
            self._condition.notify_all()

    def health(self) -> dict[str, object]:
        with self._condition:
            age = time.monotonic() - self._last_frame_time if self._last_frame_time else None
            return {
                "streaming": age is not None and age < 2.0,
                "camera_count": self._camera_count,
                "last_frame_age_s": round(age, 3) if age is not None else None,
            }

    def wait_for_jpeg(self, sequence: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence != sequence or not self._running,
                timeout=timeout,
            )
            return self._sequence, self._jpeg

    def _receive_loop(self) -> None:
        frame_period = 1.0 / max(self.config.fps, 1)
        next_frame_time = 0.0
        while self._running:
            message = self._client.receive_message_nonblocking(timeout_ms=200)
            if message is None:
                continue

            now = time.monotonic()
            if now < next_frame_time:
                continue

            decoded = ImageMessageSchema.deserialize(message)
            images = dict(decoded.images)
            depth = decoded.depths.get("ego_view_depth")
            if depth is not None:
                preview = colorize_depth(depth)
                if preview is not None:
                    images["ego_view_depth"] = preview
            jpeg = compose_camera_jpeg(
                images,
                max_tile_width=self.config.max_tile_width,
                jpeg_quality=self.config.jpeg_quality,
            )
            if jpeg is None:
                continue

            next_frame_time = now + frame_period
            with self._condition:
                self._jpeg = jpeg
                self._sequence += 1
                self._camera_count = len(images)
                self._last_frame_time = now
                self._condition.notify_all()


class RecorderControlHub:
    """Bridge HTTP requests to the recorder and cache its authoritative status."""

    def __init__(self, config: CameraWebViewerConfig):
        self.config = config
        self._commands: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._status: dict[str, object] | None = None
        self._status_received_at = 0.0
        self._running = True
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def send(self, command: str) -> None:
        if command not in {"c", "x"}:
            raise ValueError(f"unsupported recorder command: {command}")
        self._commands.put(command)

    def list_dataset_repos(self) -> list[str]:
        """Return dataset names owned by the configured Hugging Face organization."""
        prefix = f"{self.config.hf_namespace}/"
        repo_ids = [
            item.id
            for item in HfApi().list_datasets(author=self.config.hf_namespace)
            if item.id.startswith(prefix)
        ]
        return sorted(repo_ids, key=str.casefold)

    def configure_dataset(self, repo_name: str, prompt: str, private: bool) -> str:
        """Create/select an empty Hub repo and forward the session config."""
        repo_name = repo_name.strip()
        prompt = prompt.strip()
        if not isinstance(private, bool):
            raise ValueError("private must be a boolean")
        if repo_name.startswith(f"{self.config.hf_namespace}/"):
            repo_name = repo_name.split("/", 1)[1]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", repo_name):
            raise ValueError("repository name must use letters, numbers, '.', '_' or '-'")
        if not prompt:
            raise ValueError("task prompt cannot be empty")
        if len(prompt) > 1000:
            raise ValueError("task prompt must be 1000 characters or fewer")

        status = self.status()
        if status.get("recording") or status.get("saving"):
            raise RuntimeError("stop or discard the active episode first")
        finalizer = status.get("finalizer") or {}
        if finalizer.get("pending") or finalizer.get("finalizing"):
            raise RuntimeError("wait for local episode finalization before changing dataset")
        if int(status.get("total_episodes", 0) or 0) > 0:
            raise RuntimeError("repository and prompt are locked after the first saved episode")

        repo_id = f"{self.config.hf_namespace}/{repo_name}"
        api = HfApi()
        try:
            existing_files = api.list_repo_files(repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            existing_files = []
        populated = any(
            path.startswith(("data/", "videos/", "meta/")) for path in existing_files
        )
        current_repo = (status.get("hub") or {}).get("repo_id")
        if populated and current_repo != repo_id:
            raise ValueError(
                "that dataset already contains episodes; choose an empty repo or create a new one"
            )

        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.update_repo_settings(repo_id, repo_type="dataset", private=private)
        self._commands.put(encode_dataset_config(repo_id, prompt, private))
        return repo_id

    def status(self) -> dict[str, object]:
        with self._lock:
            payload = dict(self._status or {})
            age = time.monotonic() - self._status_received_at if self._status else None
        payload["connected"] = age is not None and age < 2.0
        payload["last_status_age_s"] = round(age, 3) if age is not None else None
        return payload

    def _run(self) -> None:
        context = zmq.Context()
        command_socket = context.socket(zmq.PUB)
        command_socket.setsockopt(zmq.SNDHWM, 10)
        command_socket.bind(f"tcp://*:{self.config.recording_command_port}")
        status_socket = context.socket(zmq.SUB)
        status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        status_socket.setsockopt(zmq.CONFLATE, 1)
        status_socket.connect(
            f"tcp://{self.config.recording_status_host}:{self.config.recording_status_port}"
        )
        poller = zmq.Poller()
        poller.register(status_socket, zmq.POLLIN)
        self._ready.set()
        try:
            while self._running:
                try:
                    while True:
                        command = self._commands.get_nowait()
                        command_socket.send_string(command)
                        # Configuration is idempotent. Repeat it briefly so a
                        # newly-connected SUB socket cannot miss the setup.
                        if command.startswith("dataset_config:"):
                            for _ in range(3):
                                time.sleep(0.05)
                                command_socket.send_string(command)
                except queue.Empty:
                    pass
                if status_socket in dict(poller.poll(50)):
                    payload = status_socket.recv_json()
                    with self._lock:
                        self._status = payload
                        self._status_received_at = time.monotonic()
        finally:
            command_socket.close(linger=0)
            status_socket.close(linger=0)
            context.term()


class HandControlHub:
    """Cache hand state and bridge UI reconnect requests to the worker."""

    def __init__(self, config: CameraWebViewerConfig):
        self.config = config
        self._commands: queue.Queue[int] = queue.Queue()
        self._lock = threading.Lock()
        self._status: dict[str, object] | None = None
        self._status_received_at = 0.0
        self._running = True
        self._ready = threading.Event()
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self.config.enable_hand_controls:
            return
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def close(self) -> None:
        if not self.config.enable_hand_controls:
            return
        self._running = False
        self._thread.join(timeout=2.0)

    def reconnect(self) -> None:
        if not self.config.enable_hand_controls:
            raise RuntimeError("external-hand controls are disabled")
        self._sequence += 1
        self._commands.put(self._sequence)

    def status(self) -> dict[str, object]:
        if not self.config.enable_hand_controls:
            return {"enabled": False, "connected": False}
        with self._lock:
            payload = dict(self._status or {})
            age = time.monotonic() - self._status_received_at if self._status else None
        state_age = payload.get("state_age_s")
        state_is_fresh = (
            state_age is None
            or (
                not isinstance(state_age, bool)
                and isinstance(state_age, (int, float))
                and state_age < 1.0
            )
        )
        payload["connected"] = age is not None and age < 1.0 and state_is_fresh
        payload["enabled"] = True
        payload["last_status_age_s"] = round(age, 3) if age is not None else None
        return payload

    def _run(self) -> None:
        context = zmq.Context()
        command_socket = context.socket(zmq.PUB)
        command_socket.setsockopt(zmq.SNDHWM, 10)
        command_socket.bind(f"tcp://*:{self.config.hand_control_port}")
        status_socket = context.socket(zmq.SUB)
        status_socket.setsockopt(zmq.CONFLATE, 1)
        status_socket.setsockopt(zmq.SUBSCRIBE, HAND_STATE_TOPIC)
        status_socket.connect(
            f"tcp://{self.config.hand_state_host}:{self.config.hand_state_port}"
        )
        poller = zmq.Poller()
        poller.register(status_socket, zmq.POLLIN)
        self._ready.set()
        try:
            while self._running:
                try:
                    while True:
                        sequence = self._commands.get_nowait()
                        command_socket.send(
                            encode(
                                HAND_CONTROL_TOPIC,
                                {
                                    "schema": HAND_CONTROL_SCHEMA,
                                    "sequence": sequence,
                                    "action": "reconnect",
                                    "monotonic_ns": time.monotonic_ns(),
                                    "source": "web_ui",
                                },
                            )
                        )
                except queue.Empty:
                    pass
                if status_socket in dict(poller.poll(50)):
                    try:
                        payload = decode_state(status_socket.recv())
                    except Exception as exc:
                        print(f"[Hands UI] Ignoring invalid state: {exc}")
                        continue
                    with self._lock:
                        self._status = payload
                        self._status_received_at = time.monotonic()
        finally:
            command_socket.close(linger=0)
            status_socket.close(linger=0)
            context.term()


class TeleopControlHub:
    """Serialize browser safety commands onto the manager's local ZMQ channel."""

    def __init__(self, config: CameraWebViewerConfig):
        self.config = config
        self._commands: queue.Queue[dict[str, object]] = queue.Queue()
        self._running = True
        self._ready = threading.Event()
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)

    def safe_idle(self) -> None:
        self._sequence = max(self._sequence + 1, time.monotonic_ns())
        self._commands.put(
            {
                "sequence": self._sequence,
                "action": "safe_idle",
                "monotonic_ns": time.monotonic_ns(),
                "source": "web_ui",
            }
        )

    def _run(self) -> None:
        context = zmq.Context()
        command_socket = context.socket(zmq.PUB)
        command_socket.setsockopt(zmq.SNDHWM, 10)
        command_socket.bind(f"tcp://*:{self.config.teleop_control_port}")
        self._ready.set()
        try:
            while self._running:
                try:
                    command = self._commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                # Repeat briefly so a newly connected SUB cannot miss a
                # safety command during PUB/SUB subscription propagation.
                for _ in range(3):
                    command_socket.send_json(command)
                    time.sleep(0.05)
        finally:
            command_socket.close(linger=0)
            context.term()


def compose_camera_jpeg(images: dict[str, np.ndarray], max_tile_width: int, jpeg_quality: int) -> bytes | None:
    """Label and horizontally tile camera frames into one browser-ready JPEG."""
    tiles: list[np.ndarray] = []
    for name in sorted(images):
        image = images[name]
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            continue

        if image.ndim == 2:
            tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            tile = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Camera clients expose RGB; OpenCV's encoder expects BGR.
            tile = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            continue

        height, width = tile.shape[:2]
        if width > max_tile_width:
            scale = max_tile_width / width
            tile = cv2.resize(tile, (max_tile_width, max(1, int(height * scale))))

        cv2.putText(
            tile,
            name,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (80, 230, 140),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return None

    max_height = max(tile.shape[0] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        if tile.shape[0] < max_height:
            tile = cv2.copyMakeBorder(
                tile,
                0,
                max_height - tile.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(10, 12, 15),
            )
        padded_tiles.append(tile)

    canvas = cv2.hconcat(padded_tiles)
    ok, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    return encoded.tobytes() if ok else None


def colorize_depth(depth: np.ndarray) -> np.ndarray | None:
    """Create a display-only RGB preview from a float depth map."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        return None
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return None
    low, high = np.percentile(values[valid], [2, 98])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~valid] = 0.0
    # Near objects are warm and far objects are cool; invalid pixels are black.
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def make_handler(
    frame_hub: CameraFrameHub,
    recorder_hub: RecorderControlHub,
    hand_hub: HandControlHub,
    teleop_hub: TeleopControlHub,
    pico_hub: PicoBodySubscriber | None = None,
) -> type[BaseHTTPRequestHandler]:
    class CameraWebHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes("text/html; charset=utf-8", _INDEX_HTML)
            elif path == "/pico/body":
                self._send_json(pico_hub.status() if pico_hub is not None else {"state": "waiting", "connected": False, "poses": None})
            elif path == "/healthz":
                payload = json.dumps(frame_hub.health()).encode("utf-8")
                self._send_bytes("application/json", payload)
            elif path == "/recording/status":
                payload = json.dumps(recorder_hub.status()).encode("utf-8")
                self._send_bytes("application/json", payload)
            elif path == "/datasets/repos":
                try:
                    self._send_json(
                        {
                            "namespace": recorder_hub.config.hf_namespace,
                            "repos": recorder_hub.list_dataset_repos(),
                        }
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            elif path == "/hands/status":
                payload = json.dumps(hand_hub.status()).encode("utf-8")
                self._send_bytes("application/json", payload)
            elif path == "/snapshot.jpg":
                _, jpeg = frame_hub.wait_for_jpeg(-1, timeout=2.0)
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "No camera frame yet")
                else:
                    self._send_bytes("image/jpeg", jpeg)
            elif path == "/stream.mjpg":
                self._stream_mjpeg()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = self.path.split("?", 1)[0]
            if path == "/recording/toggle":
                recorder_hub.send("c")
            elif path == "/recording/discard":
                recorder_hub.send("x")
            elif path == "/dataset/configure":
                try:
                    request = self._read_json()
                    repo_id = recorder_hub.configure_dataset(
                        str(request.get("repo", "")),
                        str(request.get("prompt", "")),
                        request.get("private", True),
                    )
                except (ValueError, RuntimeError) as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                except Exception as exc:
                    self._send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                self._send_json({"accepted": True, "repo_id": repo_id})
                return
            elif path == "/hands/reconnect":
                hand_hub.reconnect()
            elif path == "/teleop/safe-idle":
                teleop_hub.safe_idle()
            elif path == "/sonic/disconnect":
                if self.headers.get("X-Sonic-Confirmation") != "disconnect":
                    self.send_error(HTTPStatus.BAD_REQUEST, "Explicit confirmation required")
                    return
                result = subprocess.run(
                    ["tmux", "send-keys", "-t", frame_hub.config.sonic_tmux_target, "C-c"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, result.stderr.strip())
                    return
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes("application/json", b'{"accepted":true}')

        def _read_json(self) -> dict[str, object]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16_384:
                raise ValueError("invalid JSON request size")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("JSON request must be an object")
            return payload

        def _send_json(
            self,
            payload: dict[str, object],
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _send_bytes(self, content_type: str, payload: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _stream_mjpeg(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            try:
                while True:
                    sequence, jpeg = frame_hub.wait_for_jpeg(sequence)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            return

    return CameraWebHandler


def main(config: CameraWebViewerConfig) -> None:
    if config.camera_port == config.http_port and config.camera_host in {"localhost", "127.0.0.1"}:
        raise ValueError("--camera-port and --http-port must be different")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    frame_hub = CameraFrameHub(config)
    recorder_hub = RecorderControlHub(config)
    hand_hub = HandControlHub(config)
    teleop_hub = TeleopControlHub(config)
    pico_hub = PicoBodySubscriber(config.pico_body_host, config.pico_body_port)
    server = ThreadingHTTPServer(
        (config.http_host, config.http_port),
        make_handler(frame_hub, recorder_hub, hand_hub, teleop_hub, pico_hub),
    )
    server.daemon_threads = True

    def stop_server(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    frame_hub.start()
    recorder_hub.start()
    hand_hub.start()
    teleop_hub.start()
    pico_hub.start()
    print(
        f"SONIC browser viewer listening on http://{config.http_host}:{config.http_port}\n"
        "Use an SSH local port forward when viewing from another computer."
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        frame_hub.close()
        recorder_hub.close()
        hand_hub.close()
        teleop_hub.close()
        pico_hub.close()


if __name__ == "__main__":
    main(tyro.cli(CameraWebViewerConfig))
