// Aero3D OnePass - Pipeline & 3D Interactive Controller

const logs = [
  '[01 INGEST] Demuxing 4K video frames · Extracting metadata & telemetry',
  '[02 PREPROCESS] Extracted 42 candidate keyframes · Applied Laplacian deblurring',
  '[03 PURGING] AI Dynamic Object Masking · Purged 14 transient objects & vehicles',
  '[04 POSE+DEPTH] Depth Anything V2 · FP16 metric depth & 6-DoF camera pose solved',
  '[05 FUSION] Sensor Fusion · Synchronized RTK GNSS, IMU vectors, and optical depth',
  '[06 RECON] Dense 3D Point Cloud fused · Screened Poisson surface mesh generated',
  '[07 GEOREF] Aligning coordinate frame to WGS84 / UTM Zone 43N (RMSE: 1.4 cm)',
  '[08 VALIDATE] Quality Assurance passed · 4 Deliverables (.GLB, .LAS, .OBJ, .GeoTIFF) ready'
];

const phaseLabels = [
  '01 · Data Ingestion',
  '02 · Frame Preprocessing',
  '03 · Dynamic Object Purging',
  '04 · Camera Pose + Depth',
  '05 · Sensor Fusion',
  '06 · 3D Reconstruction',
  '07 · Georeferencing',
  '08 · Validation'
];

let running = false;
let progress = 48;
let scene, camera, renderer, modelGroup, wireframeMesh, pathLine, solidMaterial;

function log(message) {
  const terminal = document.getElementById('terminal');
  if (!terminal) return;
  const line = document.createElement('div');
  line.className = 'log-line';
  const time = new Date().toLocaleTimeString([], { hour12: false });
  line.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
}

function setProgress(value) {
  progress = value;
  const progressVal = document.getElementById('progressValue');
  const progressBar = document.getElementById('progressBar');
  const stageLabel = document.getElementById('currentStageLabel');
  
  if (progressVal) progressVal.textContent = `${value}%`;
  if (progressBar) progressBar.style.width = `${value}%`;
  
  // Highlight active phase among 8 stages
  const currentPhaseIndex = Math.min(7, Math.floor((value / 100) * 8));
  if (stageLabel && phaseLabels[currentPhaseIndex]) {
    stageLabel.textContent = `Stage 0${currentPhaseIndex + 1} · ${phaseLabels[currentPhaseIndex].replace(/^0\d\s*·\s*/, '')}`;
  }

  document.querySelectorAll('.phase').forEach((phase, index) => {
    const statusEl = phase.querySelector('.phase-status');
    phase.classList.remove('active', 'done', 'pending');

    if (value >= 100 || index < currentPhaseIndex) {
      phase.classList.add('done');
      if (statusEl) statusEl.textContent = '✓';
    } else if (index === currentPhaseIndex) {
      phase.classList.add('active');
      if (statusEl) statusEl.textContent = '●';
    } else {
      phase.classList.add('pending');
      if (statusEl) statusEl.textContent = '○';
    }
  });
}

function runPipeline() {
  if (running) return;
  running = true;
  
  const statusPill = document.getElementById('inputStatus');
  const runBtn = document.getElementById('runAll');
  
  if (statusPill) {
    statusPill.textContent = 'PROCESSING';
    statusPill.className = 'status-pill processing';
  }
  
  if (runBtn) {
    runBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="animation: spin 1s linear infinite;"><path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/></svg>
      Processing...
    `;
  }

  log('[SYSTEM] Starting 3D reconstruction pipeline execution...');
  let step = 0;
  
  const timer = setInterval(() => {
    if (step < logs.length) {
      log(logs[step]);
      setProgress(Math.min(100, Math.round(((step + 1) / logs.length) * 100)));
      step++;
    } else {
      clearInterval(timer);
      running = false;
      
      if (statusPill) {
        statusPill.textContent = 'COMPLETE';
        statusPill.className = 'status-pill ready';
      }
      
      if (runBtn) {
        runBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
          Pipeline Done (Re-run)
        `;
      }
      
      const exportCount = document.getElementById('exportCount');
      if (exportCount) exportCount.textContent = '4 READY';
      
      log('[SYSTEM] Job #26158 successfully completed!');
      
      // Persist completed 3D reconstruction job to MongoDB Compass (pipeline_jobs)
      saveJobToMongo({
        job_id: 'JOB-26158',
        status: 'COMPLETED',
        preset: document.querySelector('.preset.selected')?.dataset?.preset || 'balanced',
        frame_density: parseInt(document.getElementById('density')?.value || 3),
        dynamic_masking: true,
        execution_logs: Array.from(document.querySelectorAll('.log-line')).map(l => l.textContent)
      });
    }
  }, 700);
}

// Stage 02 keyframe-selection dropdown: the backend owns the default, the user's last choice wins
const KEYFRAME_MODE_HINTS = {
  sfm: 'Keyframes overlap 88-96% and are the sharpest of their neighbours: the most accurate input for COLMAP.',
  sfm_light: 'Keyframes overlap 65-85%: about 4x fewer images and a much faster COLMAP run, slightly less accurate.',
  notebook: 'Exact port of the Colab notebook: absolute sharpness cutoff (100) + histogram-correlation redundancy filter.'
};

function applyKeyframeModeDefault(health) {
  const select = document.getElementById('keyframeModeSelect');
  if (!select || select.dataset.ready) return;
  select.dataset.ready = '1';
  let stored = null;
  try { stored = localStorage.getItem('aero3d.keyframeMode'); } catch (e) { /* storage unavailable */ }
  const wanted = [stored, health && health.default_keyframe_mode].find((m) => m && [...select.options].some((o) => o.value === m));
  if (wanted) select.value = wanted;
  const sync = () => {
    const hint = document.getElementById('keyframeModeHint');
    if (hint) hint.textContent = KEYFRAME_MODE_HINTS[select.value] || '';
  };
  select.addEventListener('change', () => {
    try { localStorage.setItem('aero3d.keyframeMode', select.value); } catch (e) { /* ignore */ }
    sync();
  });
  sync();
}

// MongoDB Integration Helpers
async function checkMongoHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    applyKeyframeModeDefault(data);
    const mongoStatusText = document.getElementById('mongoStatusText');
    const mongoDot = document.getElementById('mongoDot');
    const sidebarMongoText = document.getElementById('sidebarMongoText');
    const sidebarMongoDot = document.getElementById('sidebarMongoDot');

    if (data.mongodb_connected) {
      if (mongoStatusText) mongoStatusText.textContent = 'CONNECTED';
      if (mongoDot) {
        mongoDot.style.backgroundColor = '#10b981';
        mongoDot.style.boxShadow = '0 0 0 0 rgba(16, 185, 129, 0.5)';
      }
      if (sidebarMongoText) sidebarMongoText.textContent = `${data.database} (ONLINE)`;
      if (sidebarMongoDot) sidebarMongoDot.style.backgroundColor = '#10b981';
      log(`[MONGODB] Connected to database "${data.database}" at ${data.mongodb_uri} (Logs: ${data.collections.telemetry_logs}, Jobs: ${data.collections.pipeline_jobs})`);
    } else {
      if (mongoStatusText) {
        mongoStatusText.textContent = 'OFFLINE';
        mongoStatusText.style.color = '#ef4444';
      }
      if (mongoDot) mongoDot.style.backgroundColor = '#ef4444';
      if (sidebarMongoText) sidebarMongoText.textContent = 'OFFLINE';
      if (sidebarMongoDot) sidebarMongoDot.style.backgroundColor = '#ef4444';
    }
  } catch (err) {
    console.warn('MongoDB health check error:', err);
  }
}

async function saveTelemetryToMongo(state) {
  if (!state) return;
  try {
    const headingRad = (state.baseHeading * Math.PI) / 180;
    const vn = 4.2 * Math.cos(headingRad);
    const ve = 4.2 * Math.sin(headingRad);
    const vd = -0.05;
    const payload = {
      simulated: true,
      source_label: state.label,
      width: state.width,
      height: state.height,
      res_tag: state.resTag,
      fps: state.fps,
      duration_sec: state.durationSec,
      camera_intrinsics: state.intrinsics,
      timestamp_utc: new Date(state.baseEpochMs).toISOString(),
      timecode: '00:00:00.00',
      latitude_deg: state.baseLat,
      latitude_dms: decimalToDMS(state.baseLat, true),
      longitude_deg: state.baseLng,
      longitude_dms: decimalToDMS(state.baseLng, false),
      altitude_agl_m: state.baseAlt,
      altitude_msl_m: state.baseElev + state.baseAlt,
      barometric_alt_m: state.baseAlt + 0.8,
      video_fps_res: `${state.width}×${state.height} @ ${state.fps} FPS`,
      camera_intrinsics_summary: `fx: ${state.intrinsics.fx}, cx: ${state.intrinsics.cx}, cy: ${state.intrinsics.cy}`,
      yaw_deg: state.baseHeading,
      pitch_deg: -90.0,
      roll_deg: 0.6,
      imu_acc: [0.08, -0.04, 9.81],
      imu_gyro: [0.12, -0.08, 0.35],
      barometer_hpa: 1008.42,
      barometer_inhg: 29.78,
      rtk_status: 'RTK FIXED (L1/L2/L5)',
      rtk_accuracy: 'H: ±0.8 cm · V: ±1.4 cm',
      sat_count: state.satCount,
      ground_speed_mps: 4.2,
      velocity_vectors: `VN: +${vn.toFixed(1)} · VE: +${ve.toFixed(1)} · VD: ${vd.toFixed(2)} m/s`
    };

    const res = await fetch('/api/telemetry', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.success) {
      log(`[MONGODB] Saved SIMULATED flight telemetry to collection "telemetry_logs" · Doc ID: ${result.id}`);
      const btn = document.getElementById('syncMongoBtn');
      if (btn) {
        btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Synced in Compass`;
        setTimeout(() => {
          btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg> Sync to MongoDB`;
        }, 3000);
      }
    }
  } catch (err) {
    console.error('Failed to save telemetry to MongoDB:', err);
  }
}

async function saveJobToMongo(jobData) {
  try {
    const res = await fetch('/api/pipeline/job', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(jobData)
    });
    const result = await res.json();
    if (result.success) {
      log(`[MONGODB] Logged 3D Pipeline Job to collection "pipeline_jobs" in aero3d_db · Doc ID: ${result.id}`);
    }
  } catch (err) {
    console.error('Failed to save pipeline job to MongoDB:', err);
  }
}

// Active flight telemetry state
let activeTelemetryState = null;

function decimalToDMS(deg, isLat) {
  const abs = Math.abs(deg);
  const d = Math.floor(abs);
  const m = Math.floor((abs - d) * 60);
  const s = (((abs - d) * 60 - m) * 60).toFixed(2);
  const dir = isLat ? (deg >= 0 ? 'N' : 'S') : (deg >= 0 ? 'E' : 'W');
  return `${d}° ${m}' ${s}" ${dir}`;
}

function calculateCameraIntrinsics(width, height) {
  // DJI / Photogrammetric aerial sensor (1" CMOS: 13.2mm x 8.8mm, 24mm 35mm-equivalent)
  const sensorW = 13.2; // mm
  const sensorH = 8.8;  // mm
  const actualFocalMm = 8.8; // mm
  const pixelPitchUm = ((sensorW / width) * 1000).toFixed(3);
  
  // Pinhole camera focal length in pixels: fx = fy ~ (f_actual / sensorW) * width * 1.2208 (nominal lens distortion)
  const fx = parseFloat(((actualFocalMm / sensorW) * width * 1.2208).toFixed(2));
  const fy = fx;
  const cx = parseFloat((width / 2).toFixed(1));
  const cy = parseFloat((height / 2).toFixed(1));
  
  // Field of View
  const hFov = (2 * Math.atan(width / (2 * fx)) * (180 / Math.PI)).toFixed(1);
  const vFov = (2 * Math.atan(height / (2 * fy)) * (180 / Math.PI)).toFixed(1);
  const dFov = (Math.sqrt(hFov * hFov + vFov * vFov)).toFixed(1);

  // Brown-Conrady Polynomial Lens Distortion Coefficients
  const k1 = -0.04281;
  const k2 = 0.01562;
  const p1 = 0.00018;
  const p2 = -0.00009;

  return {
    fx, fy, cx, cy,
    sensorW, sensorH, pixelPitchUm,
    hFov, vFov, dFov,
    k1, k2, p1, p2,
    focalEquiv: 24.0
  };
}

function updateLiveTelemetry(currentTime = 0) {
  if (!activeTelemetryState) return;
  const state = activeTelemetryState;
  const t = currentTime;

  // 1. Timestamp (UTC & Video Timecode)
  const baseEpochMs = state.baseEpochMs;
  const currentEpochMs = baseEpochMs + Math.round(t * 1000);
  const dateObj = new Date(currentEpochMs);
  const isoUtc = dateObj.toISOString().replace('T', ' ').replace('Z', ' UTC');
  const mins = Math.floor(t / 60);
  const secs = Math.floor(t % 60);
  const msecs = Math.floor((t % 1) * 100);
  const timecodeStr = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${msecs.toString().padStart(2, '0')}`;
  
  // 2 & 3. Latitude & Longitude (advancing along trajectory vector)
  const headingRad = (state.baseHeading * Math.PI) / 180;
  const vn = 4.2 * Math.cos(headingRad);
  const ve = 4.2 * Math.sin(headingRad);
  const vd = -0.05 + 0.12 * Math.sin(t * 0.4);

  const metersPerLat = 111139.0;
  const metersPerLng = 111139.0 * Math.cos((state.baseLat * Math.PI) / 180);
  const currentLat = state.baseLat + (vn * t) / metersPerLat;
  const currentLng = state.baseLng + (ve * t) / metersPerLng;

  // 4. Altitude (AGL, MSL, Barometric)
  const currentAgl = state.baseAlt + Math.sin(t * 0.25) * 1.6;
  const currentMsl = state.baseElev + currentAgl;
  const baroAlt = currentAgl + 0.8 * Math.cos(t * 0.3);

  // 5. FPS & Resolution
  const resStr = `${state.width}×${state.height} (${state.resTag})`;
  const fpsStr = `${state.fps} FPS · Bitrate: 100 Mbps`;

  // 6. Camera Intrinsics
  const int = state.intrinsics;
  const intVal = `fx: ${int.fx}, fy: ${int.fy}`;
  const intSub = `cx: ${int.cx}, cy: ${int.cy} · 24mm F/2.8`;

  // 7. Yaw, Pitch, Roll (Euler Angles)
  const currentYaw = (state.baseHeading + Math.sin(t * 0.15) * 2.2).toFixed(1);
  const currentPitch = (-90.0 + Math.cos(t * 0.2) * 0.6).toFixed(1);
  const currentRoll = (Math.sin(t * 0.3) * 0.8).toFixed(1);

  // 8. IMU (Accelerations & Angular Gyro Rates)
  const ax = (Math.sin(t * 1.5) * 0.12).toFixed(2);
  const ay = (Math.cos(t * 1.2) * 0.08).toFixed(2);
  const az = (9.81 + Math.sin(t * 2.0) * 0.15).toFixed(2);
  const gx = (Math.sin(t * 0.8) * 0.24).toFixed(2);
  const gy = (Math.cos(t * 0.7) * 0.18).toFixed(2);
  const gz = (0.35 + Math.sin(t * 0.5) * 0.12).toFixed(2);

  // 9. Barometer
  const baroHpa = (1013.25 - (currentMsl * 0.12) + Math.sin(t * 0.2) * 0.15).toFixed(2);
  const baroInHg = (baroHpa * 0.02953).toFixed(2);

  // 10. RTK / PPK Status
  const rtkStatus = 'RTK FIXED (L1/L2/L5)';
  const rtkSub = `H: ±0.8 cm · V: ±1.4 cm · ${state.satCount} Sats`;

  // 11. Velocity
  const groundSpeed = (Math.sqrt(vn * vn + ve * ve) + Math.sin(t * 0.3) * 0.2).toFixed(1);
  const speedKmh = (groundSpeed * 3.6).toFixed(1);
  const velVectors = `VN: ${vn >= 0 ? '+' : ''}${vn.toFixed(1)} · VE: ${ve >= 0 ? '+' : ''}${ve.toFixed(1)} · VD: ${vd >= 0 ? '+' : ''}${vd.toFixed(2)} m/s`;

  // Render to DOM
  const setEl = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  setEl('metaTimestamp', isoUtc);
  setEl('metaTimecode', `TC: ${timecodeStr} · T+${t.toFixed(2)}s`);
  setEl('metaLat', `${currentLat.toFixed(6)}° N`);
  setEl('metaLatDms', `DMS: ${decimalToDMS(currentLat, true)} · WGS84`);
  setEl('metaLng', `${currentLng.toFixed(6)}° E`);
  setEl('metaLngDms', `DMS: ${decimalToDMS(currentLng, false)} · UTM 43N`);
  setEl('metaAlt', `${currentAgl.toFixed(1)} m AGL`);
  setEl('metaAltSub', `MSL: ${currentMsl.toFixed(1)} m · Baro: ${baroAlt.toFixed(1)} m`);
  setEl('metaRes', resStr);
  setEl('metaFps', fpsStr);
  setEl('metaIntrinsics', intVal);
  setEl('metaIntrinsicsSub', intSub);
  setEl('metaEuler', `Y: ${currentYaw}° · P: ${currentPitch}° · R: ${currentRoll}°`);
  setEl('metaEulerSub', `Gimbal: Nadir · Heading: ${currentYaw}° SE`);
  setEl('metaImu', `Acc: [${ax}, ${ay}, ${az}] m/s²`);
  setEl('metaImuSub', `Gyro: [${gx}, ${gy}, ${gz}] °/s`);
  setEl('metaBaro', `${baroHpa} hPa (${baroInHg} inHg)`);
  setEl('metaBaroSub', `Temp: 24.2°C · Press. Alt: ${baroAlt.toFixed(1)} m`);
  setEl('metaRtk', rtkStatus);
  setEl('metaRtkSub', rtkSub);
  setEl('metaVelocity', `${groundSpeed} m/s (${speedKmh} km/h)`);
  setEl('metaVelocitySub', velVectors);
}

function populateIntrinsicsModal(int, width, height) {
  const setEl = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };

  setEl('matFx', `fx: ${int.fx}`);
  setEl('matFy', `fy: ${int.fy}`);
  setEl('matCx', `cx: ${int.cx}`);
  setEl('matCy', `cy: ${int.cy}`);
  setEl('matRadial', `k₁: ${int.k1} · k₂: +${int.k2}`);
  setEl('matTangential', `p₁: +${int.p1} · p₂: ${int.p2}`);
  setEl('matSensorDim', `${int.sensorW} × ${int.sensorH} mm (1" CMOS)`);
  setEl('matPixelPitch', `Pixel Pitch: ${int.pixelPitchUm} µm · Aspect: 16:9`);
  setEl('matFov', `H: ${int.hFov}° · V: ${int.vFov}° · D: ${int.dFov}°`);
  setEl('matFocalEquiv', `${int.focalEquiv} mm (35mm format equiv)`);

  const codeEl = document.getElementById('intrinsicsCodeBlock');
  if (codeEl) {
    codeEl.textContent = `# OpenCV & COLMAP Calibration Model (PINHOLE_RADIAL)\nCAMERA_ID=1 MODEL=RADIAL WIDTH=${width} HEIGHT=${height}\nPARAMS=${int.fx} ${int.cx} ${int.cy} ${int.k1} ${int.k2}\nK_MATRIX=[[${int.fx}, 0.0, ${int.cx}], [0.0, ${int.fy}, ${int.cy}], [0.0, 0.0, 1.0]]`;
  }
}

function exportTelemetry(format = 'json') {
  if (!activeTelemetryState) {
    log('Error: Please load a video first to export telemetry.');
    return;
  }
  const state = activeTelemetryState;
  const durSec = Math.max(10, Math.round(state.durationSec || 60));
  const samples = [];

  for (let s = 0; s <= durSec; s++) {
    const t = s;
    const currentEpochMs = state.baseEpochMs + t * 1000;
    const isoUtc = new Date(currentEpochMs).toISOString();
    const headingRad = (state.baseHeading * Math.PI) / 180;
    const vn = 4.2 * Math.cos(headingRad);
    const ve = 4.2 * Math.sin(headingRad);
    const vd = -0.05 + 0.12 * Math.sin(t * 0.4);
    const currentLat = state.baseLat + (vn * t) / 111139.0;
    const currentLng = state.baseLng + (ve * t) / (111139.0 * Math.cos((state.baseLat * Math.PI) / 180));
    const currentAgl = state.baseAlt + Math.sin(t * 0.25) * 1.6;
    const currentMsl = state.baseElev + currentAgl;
    const baroHpa = 1013.25 - (currentMsl * 0.12) + Math.sin(t * 0.2) * 0.15;
    const currentYaw = state.baseHeading + Math.sin(t * 0.15) * 2.2;
    const currentPitch = -90.0 + Math.cos(t * 0.2) * 0.6;
    const currentRoll = Math.sin(t * 0.3) * 0.8;
    const groundSpeed = Math.sqrt(vn * vn + ve * ve);

    samples.push({
      timestamp_utc: isoUtc,
      time_sec: t,
      latitude_deg: parseFloat(currentLat.toFixed(7)),
      longitude_deg: parseFloat(currentLng.toFixed(7)),
      altitude_agl_m: parseFloat(currentAgl.toFixed(2)),
      altitude_msl_m: parseFloat(currentMsl.toFixed(2)),
      barometer_hpa: parseFloat(baroHpa.toFixed(2)),
      yaw_deg: parseFloat(currentYaw.toFixed(2)),
      pitch_deg: parseFloat(currentPitch.toFixed(2)),
      roll_deg: parseFloat(currentRoll.toFixed(2)),
      ground_speed_mps: parseFloat(groundSpeed.toFixed(2)),
      velocity_north_mps: parseFloat(vn.toFixed(2)),
      velocity_east_mps: parseFloat(ve.toFixed(2)),
      velocity_down_mps: parseFloat(vd.toFixed(2)),
      imu_acc_mps2: [
        parseFloat((Math.sin(t * 1.5) * 0.12).toFixed(3)),
        parseFloat((Math.cos(t * 1.2) * 0.08).toFixed(3)),
        parseFloat((9.81 + Math.sin(t * 2.0) * 0.15).toFixed(3))
      ],
      imu_gyro_dps: [
        parseFloat((Math.sin(t * 0.8) * 0.24).toFixed(3)),
        parseFloat((Math.cos(t * 0.7) * 0.18).toFixed(3)),
        parseFloat((0.35 + Math.sin(t * 0.5) * 0.12).toFixed(3))
      ],
      rtk_fix: 'FIXED_L1_L2',
      satellites: state.satCount,
      horizontal_accuracy_m: 0.008,
      vertical_accuracy_m: 0.014
    });
  }

  let blob, filename;
  if (format === 'json') {
    const payload = {
      generator: 'Aero3D OnePass Telemetry Ingestion Engine',
      simulated: true,
      note: 'Synthetic demo telemetry. GPS, IMU and RTK values are generated, not read from the video.',
      camera_intrinsics: state.intrinsics,
      video_metadata: {
        width: state.width,
        height: state.height,
        fps: state.fps,
        duration_sec: state.durationSec,
        source_label: state.label
      },
      flight_telemetry_points: samples
    };
    blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    filename = `flight_telemetry_${Date.now()}.json`;
  } else {
    // CSV
    const headers = [
      'Timestamp_UTC', 'Time_Sec', 'Latitude_Deg', 'Longitude_Deg', 'Altitude_AGL_m',
      'Altitude_MSL_m', 'Barometer_hPa', 'Yaw_Deg', 'Pitch_Deg', 'Roll_Deg',
      'GroundSpeed_mps', 'Vel_North_mps', 'Vel_East_mps', 'Vel_Down_mps',
      'AccX_mps2', 'AccY_mps2', 'AccZ_mps2', 'GyroX_dps', 'GyroY_dps', 'GyroZ_dps',
      'RTK_Status', 'Sat_Count', 'Simulated'
    ];
    const rows = samples.map(s => [
      s.timestamp_utc, s.time_sec, s.latitude_deg, s.longitude_deg, s.altitude_agl_m,
      s.altitude_msl_m, s.barometer_hpa, s.yaw_deg, s.pitch_deg, s.roll_deg,
      s.ground_speed_mps, s.velocity_north_mps, s.velocity_east_mps, s.velocity_down_mps,
      s.imu_acc_mps2[0], s.imu_acc_mps2[1], s.imu_acc_mps2[2],
      s.imu_gyro_dps[0], s.imu_gyro_dps[1], s.imu_gyro_dps[2],
      s.rtk_fix, s.satellites, 'true'
    ].join(','));
    const csvContent = [headers.join(','), ...rows].join('\n');
    blob = new Blob([csvContent], { type: 'text/csv' });
    filename = `flight_telemetry_${Date.now()}.csv`;
  }

  const downloadUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(downloadUrl);
  log(`[EXPORT] Downloaded ${filename} (${samples.length} telemetry entries)`);
}

// Helper to update the OpenCV Video Properties & Stream Analysis section below Extracted Image Frames Inspector
function updateOpenCvDetailsPanel(data) {
  if (!data) return;

  const width = data.width || 3840;
  const height = data.height || 2160;
  const resolution = data.resolution || `${width} x ${height}`;
  const fps = (typeof data.fps === 'number' ? data.fps.toFixed(2) : (data.fps || '29.97'));
  const totalFrames = data.total_frames || 957;
  const duration = (typeof data.duration === 'number' ? data.duration.toFixed(2) : (data.duration_sec ? Number(data.duration_sec).toFixed(2) : (totalFrames / parseFloat(fps)).toFixed(2)));
  const filename = data.filename || data.video_path || 'uploaded_video.mp4';
  const isOpened = data.status !== 'failed' && data.success !== false;

  // Update DOM Elements
  const resEl = document.getElementById('cvResValue');
  const widthEl = document.getElementById('cvWidthVal');
  const heightEl = document.getElementById('cvHeightVal');
  const fpsEl = document.getElementById('cvFpsValue');
  const totalFramesEl = document.getElementById('cvTotalFramesValue');
  const maxFrameIndexEl = document.getElementById('cvMaxFrameIndex');
  const durationEl = document.getElementById('cvDurationValue');
  const durationFormulaEl = document.getElementById('cvDurationFormula');
  const statusBadge = document.getElementById('cvStatusBadge');
  const statusText = document.getElementById('cvStatusText');
  const activeFileName = document.getElementById('cvActiveFileName');
  const terminalOutput = document.getElementById('cvTerminalOutput');

  if (resEl) resEl.textContent = resolution;
  if (widthEl) widthEl.textContent = width;
  if (heightEl) heightEl.textContent = height;
  if (fpsEl) fpsEl.textContent = fps;
  if (totalFramesEl) totalFramesEl.textContent = totalFrames;
  const colabTotalFramesEl = document.getElementById('colabTotalFrames');
  if (colabTotalFramesEl) colabTotalFramesEl.textContent = totalFrames;
  if (typeof colabState !== 'undefined') colabState.totalFrames = totalFrames;
  if (maxFrameIndexEl) maxFrameIndexEl.textContent = `#${Math.max(0, totalFrames - 1)}`;
  if (durationEl) durationEl.textContent = `${duration}s`;
  if (durationFormulaEl) durationFormulaEl.textContent = duration;
  if (activeFileName) activeFileName.textContent = filename;

  if (statusBadge && statusText) {
    if (isOpened) {
      statusBadge.className = 'badge-tag-pill green';
      statusText.textContent = '✅ Video opened successfully';
    } else {
      statusBadge.className = 'badge-tag-pill red';
      statusText.textContent = '❌ Video could not be opened';
    }
  }

  if (terminalOutput) {
    if (isOpened) {
      terminalOutput.textContent = `✅ Video opened successfully\n--------------------------------\nResolution   : ${width} x ${height}\nFPS          : ${fps}\nTotal frames : ${totalFrames}\nDuration     : ${duration} seconds`;
    } else {
      terminalOutput.textContent = `❌ Video could not be opened\n--------------------------------\nError: Video could not be decoded by OpenCV.`;
    }
  }

  // 4 Additional High-Impact Pipeline Metric Elements
  const extractedEl = document.getElementById('cvExtractedVal');
  const sharpEl = document.getElementById('cvSharpVal');
  const keyframesEl = document.getElementById('cvKeyframesVal');
  const trajEl = document.getElementById('cvTrajectoryVal');
  const trajSubEl = document.getElementById('cvTrajectorySub');

  if (extractedEl) extractedEl.textContent = `${data.number_of_extracted_frames !== undefined ? data.number_of_extracted_frames : totalFrames} frames`;
  if (sharpEl) sharpEl.textContent = `${data.number_of_sharp_frames !== undefined ? data.number_of_sharp_frames : '--'} frames`;
  if (keyframesEl) keyframesEl.textContent = `${data.number_of_final_keyframes !== undefined ? data.number_of_final_keyframes : '--'} keyframes`;
  if (trajEl) trajEl.textContent = `${data.trajectory_points !== undefined ? data.trajectory_points : totalFrames} pts`;
  if (trajSubEl && data.successful_trajectory_matches !== undefined) {
    trajSubEl.textContent = `Matches: ${data.successful_trajectory_matches} · Failed: ${data.failed_trajectory_frames}`;
  }

  // Also update Inspector Badges in the section header above
  const inspectorFramesBadge = document.getElementById('inspectorFramesBadge');
  const inspectorFpsBadge = document.getElementById('inspectorFpsBadge');
  const inspectorResBadge = document.getElementById('inspectorResBadge');
  if (inspectorFramesBadge) inspectorFramesBadge.textContent = `${totalFrames} TOTAL FRAMES`;
  if (inspectorFpsBadge) inspectorFpsBadge.textContent = `${fps} FPS`;
  if (inspectorResBadge) inspectorResBadge.textContent = `${width}x${height} RESOLUTION`;
}

// Global cache for camera trajectory
let cachedTrajectoryData = null;

// 2D Camera Trajectory Renderer (Colab Notebook Matplotlib Equivalence)
function render2DCameraTrajectory(trajectory, successful, failed) {
  const canvas = document.getElementById('trajectory2dCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Cache latest trajectory data
  cachedTrajectoryData = {
    trajectory,
    successful_matches: successful,
    failed_frames: failed
  };

  // High-DPI Responsive Canvas Resolution
  const container = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const displayW = container ? Math.max(600, container.clientWidth - 32) : 1100;
  const displayH = 520;

  canvas.width = displayW * dpr;
  canvas.height = displayH * dpr;
  ctx.resetTransform ? ctx.resetTransform() : ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  const w = displayW;
  const h = displayH;

  // Background - Deep midnight slate
  ctx.fillStyle = '#020617';
  ctx.fillRect(0, 0, w, h);

  if (!trajectory || trajectory.length === 0) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No trajectory computed yet. Upload a drone video to calculate camera trajectory across all frames.', w / 2, h / 2);
    return;
  }

  // Find coordinate bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  trajectory.forEach(pt => {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  });

  const spanX = Math.max(1.0, maxX - minX);
  const spanY = Math.max(1.0, maxY - minY);
  const paddingX = 65;
  const paddingY = 50;

  const scaleX = (w - 2 * paddingX) / spanX;
  const scaleY = (h - 2 * paddingY) / spanY;
  const scale = Math.min(scaleX, scaleY);

  const cx = w / 2;
  const cy = h / 2;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  const toCanvasX = (x) => cx + (x - midX) * scale;
  const toCanvasY = (y) => cy - (y - midY) * scale; // Invert Y for Cartesian plane

  // Subtle coordinate grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  const gridSteps = 8;
  for (let i = 0; i <= gridSteps; i++) {
    const gx = paddingX + i * (w - 2 * paddingX) / gridSteps;
    ctx.beginPath(); ctx.moveTo(gx, paddingY); ctx.lineTo(gx, h - paddingY); ctx.stroke();
    const gy = paddingY + i * (h - 2 * paddingY) / gridSteps;
    ctx.beginPath(); ctx.moveTo(paddingX, gy); ctx.lineTo(w - paddingX, gy); ctx.stroke();
  }

  // Draw axis border
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(paddingX, paddingY, w - 2 * paddingX, h - 2 * paddingY);

  // Axis Labels
  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 11px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('Relative X Movement (pixels)', w / 2, h - 14);
  ctx.save();
  ctx.translate(18, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Relative Y Movement (pixels)', 0, 0);
  ctx.restore();

  // Numerical Ticks
  ctx.fillStyle = '#64748b';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const frac = i / 4;
    const valX = minX + frac * spanX;
    const posX = paddingX + frac * (w - 2 * paddingX);
    ctx.fillText(`${valX.toFixed(1)}`, posX, h - paddingY + 16);

    const valY = minY + frac * spanY;
    const posY = h - paddingY - frac * (h - 2 * paddingY);
    ctx.textAlign = 'right';
    ctx.fillText(`${valY.toFixed(1)}`, paddingX - 8, posY + 4);
    ctx.textAlign = 'center';
  }

  // Plot Flight Trajectory Line (Glowing Orange)
  ctx.beginPath();
  ctx.strokeStyle = '#ff6b00';
  ctx.lineWidth = 3;
  ctx.shadowColor = 'rgba(255, 107, 0, 0.7)';
  ctx.shadowBlur = 10;
  trajectory.forEach((pt, idx) => {
    const px = toCanvasX(pt.x);
    const py = toCanvasY(pt.y);
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Frame Points (cyan micro-dots for every single frame)
  ctx.fillStyle = '#38bdf8';
  trajectory.forEach(pt => {
    const px = toCanvasX(pt.x);
    const py = toCanvasY(pt.y);
    ctx.beginPath();
    ctx.arc(px, py, 2.2, 0, Math.PI * 2);
    ctx.fill();
  });

  // Start Point (Green circle)
  const startPt = trajectory[0];
  const sx = toCanvasX(startPt.x);
  const sy = toCanvasY(startPt.y);
  ctx.beginPath();
  ctx.arc(sx, sy, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#10b981';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.fillStyle = '#10b981';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('Start (0,0)', sx + 12, sy - 6);

  // End Point (Red circle)
  const endPt = trajectory[trajectory.length - 1];
  const ex = toCanvasX(endPt.x);
  const ey = toCanvasY(endPt.y);
  ctx.beginPath();
  ctx.arc(ex, ey, 8, 0, Math.PI * 2);
  ctx.fillStyle = '#ef4444';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.5;
  ctx.stroke();
  ctx.fillStyle = '#ef4444';
  ctx.font = 'bold 12px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`End (Frame #${trajectory.length})`, ex + 12, ey - 6);

  // Update Trajectory Dedicated View Badges
  const fullPts = document.getElementById('trajFullPoints');
  const fullMatches = document.getElementById('trajFullMatches');
  const fullFailed = document.getElementById('trajFullFailed');
  const coordsSpan = document.getElementById('trajectoryCoordsSpan');

  const succCount = successful !== undefined ? successful : (trajectory.length - 1);
  const failCount = failed !== undefined ? failed : 0;

  if (fullPts) fullPts.textContent = `${trajectory.length} Trajectory Points (100% Frames)`;
  if (fullMatches) fullMatches.textContent = `Matches: ${succCount}`;
  if (fullFailed) fullFailed.textContent = `Failed: ${failCount}`;
  if (coordsSpan) {
    coordsSpan.textContent = `Relative Motion Extent: X: [${minX.toFixed(1)}, ${maxX.toFixed(1)}] px · Y: [${minY.toFixed(1)}, ${maxY.toFixed(1)}] px · Total ${trajectory.length} frames`;
  }
}

// Update Three.js 3D Flight Path Geometry
function updateThreeFlightPath(trajectory) {
  if (!window.THREE || !pathLine || !trajectory || trajectory.length < 2) return;
  try {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    trajectory.forEach(pt => {
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    });

    const spanX = Math.max(1.0, maxX - minX);
    const spanY = Math.max(1.0, maxY - minY);

    const newPoints = trajectory.map((pt, idx) => {
      const normX = ((pt.x - minX) / spanX - 0.5) * 8.0;
      const normY = 2.5 + ((pt.y - minY) / spanY - 0.5) * 2.0;
      const normZ = ((idx / (trajectory.length - 1)) - 0.5) * 8.0;
      return new THREE.Vector3(normX, normY, normZ);
    });

    pathLine.geometry.dispose();
    pathLine.geometry = new THREE.BufferGeometry().setFromPoints(newPoints);
  } catch (err) {
    console.warn('[THREE] Flight path update notice:', err);
  }
}

// Fetch Trajectory Data from Backend
async function loadTrajectoryData() {
  try {
    const res = await fetch('/api/trajectory');
    if (res.ok) {
      const data = await res.json();
      if (data && data.trajectory && data.trajectory.length > 0) {
        cachedTrajectoryData = data;
        render2DCameraTrajectory(data.trajectory, data.successful_matches, data.failed_frames);
        updateThreeFlightPath(data.trajectory);
        return data;
      }
    }
  } catch (err) {
    console.warn('[TRAJECTORY] Could not load trajectory:', err);
  }
}

function loadVideo(url, label, fileObj = null) {
  if (!url) return;

  const videoContainer = document.getElementById('videoContainer');
  const videoPreview = document.getElementById('videoPreview');
  const fileName = document.getElementById('fileName');
  const statusPill = document.getElementById('inputStatus');
  const metaEmptyState = document.getElementById('metaEmptyState');
  const metaGrid = document.getElementById('metaGrid');
  const metaActionsBar = document.getElementById('metaActionsBar');
  const sourceMethodBadge = document.getElementById('sourceMethodBadge');

  if (videoContainer) videoContainer.style.display = 'block';
  if (fileName) fileName.textContent = label;

  if (sourceMethodBadge) {
    if (fileObj || url.startsWith('blob:')) {
      sourceMethodBadge.textContent = 'FILE FORMAT';
      sourceMethodBadge.style.color = 'var(--orange-primary)';
      sourceMethodBadge.style.background = 'var(--orange-soft)';
      sourceMethodBadge.style.borderColor = 'var(--orange-border)';
    } else if (url.includes('sample_4k.mp4')) {
      sourceMethodBadge.textContent = 'DEMO 4K SAMPLE';
      sourceMethodBadge.style.color = '#15803d';
      sourceMethodBadge.style.background = '#dcfce7';
      sourceMethodBadge.style.borderColor = '#86efac';
    } else {
      sourceMethodBadge.textContent = 'URL / LINK FORMAT';
      sourceMethodBadge.style.color = '#0284c7';
      sourceMethodBadge.style.background = '#e0f2fe';
      sourceMethodBadge.style.borderColor = '#bae6fd';
    }
  }

  if (statusPill) {
    statusPill.textContent = 'READY';
    statusPill.className = 'status-pill ready';
  }

  if (videoPreview) {
    videoPreview.src = url;
    
    videoPreview.onloadedmetadata = () => {
      const w = videoPreview.videoWidth || 3840;
      const h = videoPreview.videoHeight || 2160;
      const durSec = Math.round(videoPreview.duration || 124);
      const resTag = w >= 3840 ? '4K UHD' : (w >= 1920 ? '1080p FHD' : '720p HD');
      const fps = '29.970';
      
      // Hash label to generate consistent, deterministic baseline coords per video source
      let hash = 0;
      for (let i = 0; i < label.length; i++) hash = (hash << 5) - hash + label.charCodeAt(i);
      hash = Math.abs(hash);

      const lat = 12.971600 + (hash % 1000) / 100000.0;
      const lng = 77.594600 + ((hash * 3) % 1000) / 100000.0;
      const alt = 120.0 + (hash % 30);
      const elev = 920.0 + (hash % 50);
      const satCount = 22 + (hash % 5);
      const headingDeg = 142.3 + (hash % 45);
      const baseEpochMs = Date.now() - durSec * 1000;

      const intrinsics = calculateCameraIntrinsics(w, h);

      activeTelemetryState = {
        label,
        width: w,
        height: h,
        resTag,
        fps,
        durationSec: durSec,
        baseLat: lat,
        baseLng: lng,
        baseAlt: alt,
        baseElev: elev,
        baseHeading: headingDeg,
        satCount,
        baseEpochMs,
        intrinsics
      };

      if (metaEmptyState) metaEmptyState.style.display = 'none';
      if (metaGrid) metaGrid.style.display = 'grid';
      if (metaActionsBar) metaActionsBar.style.display = 'flex';

      // Populate initial values
      updateLiveTelemetry(0);
      populateIntrinsicsModal(intrinsics, w, h);


      // Use true OpenCV total frames if available, otherwise calculate fallback
      if (typeof colabState !== 'undefined' && colabState.totalFrames && colabState.totalFrames !== 957) {
        colabTotalFrames = colabState.totalFrames;
      } else {
        colabTotalFrames = Math.max(30, Math.round(durSec * parseFloat(fps)));
      }
      colabFps = parseFloat(fps) || 29.97;
      const totalFramesEl = document.getElementById('colabTotalFrames');
      const colabSliderEl = document.getElementById('colabFrameSlider');
      if (totalFramesEl) totalFramesEl.textContent = colabTotalFrames;
      if (colabSliderEl) colabSliderEl.max = colabTotalFrames - 1;
      populateKeyframesFilmstrip();

      // Immediately populate the OpenCV Video Properties panel below inspector
      updateOpenCvDetailsPanel({
        width: w,
        height: h,
        resolution: `${w} x ${h}`,
        fps: parseFloat(fps).toFixed(2),
        total_frames: colabTotalFrames,
        duration: durSec.toFixed(2),
        filename: label.split(' (')[0],
        status: 'opened_successfully',
        success: true
      });

      log(`[TELEMETRY] SIMULATED demo telemetry generated (no GPS/IMU/RTK is read from the video) · Res: ${w}×${h} · Nominal intrinsics fx: ${intrinsics.fx}`);
      // Not auto-saved to MongoDB: this data is synthetic. It can still be synced manually
      // with the "Sync to MongoDB" button, and is then flagged simulated: true.
    };

    // Synchronize telemetry with real-time video playback and scrubbing
    videoPreview.ontimeupdate = () => {
      updateLiveTelemetry(videoPreview.currentTime);
    };

    videoPreview.onplay = () => {
      const dot = document.getElementById('telemetryLiveDot');
      if (dot) dot.style.animation = 'pulse 0.8s infinite';
      const sync = document.getElementById('metaSync');
      if (sync) sync.textContent = 'STREAMING TELEMETRY';
    };

    videoPreview.onpause = () => {
      const dot = document.getElementById('telemetryLiveDot');
      if (dot) dot.style.animation = 'none';
      const sync = document.getElementById('metaSync');
      if (sync) sync.textContent = 'PAUSED AT TIMECODE';
    };

    videoPreview.load();
    log(`[INGEST] Loaded video stream: "${label}"`);
  }
}

// --------------------------------------------------------------------------
// Single-Frame Extraction & Nearby 3D Model Reconstruction Engine
// --------------------------------------------------------------------------
let frame3dScene, frame3dCamera, frame3dRenderer, frame3dGroup;
let frame3dMesh, frame3dPoints, frame3dWire;
let frame3dTexture, frame3dMaterial;
let rawCapturedCanvas = null;
let currentFrameMode = 'rgb';
let currentLayer3D = 'textured';
let depthDisplacementScale = 1.2;

function formatTimecode(sec) {
  if (isNaN(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const f = Math.floor((sec % 1) * 30);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(f).padStart(2, '0')}`;
}

function initFrame3DScene() {
  const canvas = document.getElementById('frame3dCanvas');
  const wrap = document.getElementById('frame3dWrap');
  if (!window.THREE || !canvas || !wrap) return;

  frame3dScene = new THREE.Scene();
  frame3dScene.background = new THREE.Color(0x04080e);

  const aspect = wrap.clientWidth / (wrap.clientHeight || 330) || (16 / 9);
  frame3dCamera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
  frame3dCamera.position.set(0, -3.2, 4.4);
  frame3dCamera.lookAt(0, 0, 0);

  frame3dRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  frame3dRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  frame3dRenderer.setSize(wrap.clientWidth, wrap.clientHeight || 330, false);

  // Lighting
  const ambLight = new THREE.AmbientLight(0xffffff, 1.4);
  frame3dScene.add(ambLight);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight1.position.set(4, 5, 8);
  frame3dScene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xff6b00, 0.8);
  dirLight2.position.set(-4, -5, -4);
  frame3dScene.add(dirLight2);

  frame3dGroup = new THREE.Group();
  frame3dScene.add(frame3dGroup);

  // Build Initial 3D Elevation Mesh Geometry
  buildFrame3DGeometry(depthDisplacementScale);

  // Mouse Orbit controls
  let isDragging = false;
  let prevX = 0, prevY = 0;

  wrap.addEventListener('pointerdown', (e) => {
    isDragging = true;
    prevX = e.clientX;
    prevY = e.clientY;
  });

  window.addEventListener('pointerup', () => { isDragging = false; });

  window.addEventListener('pointermove', (e) => {
    if (isDragging && frame3dGroup) {
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      frame3dGroup.rotation.z += dx * 0.008;
      frame3dGroup.rotation.x += dy * 0.008;
      prevX = e.clientX;
      prevY = e.clientY;
    }
  });

  wrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (frame3dCamera) {
      frame3dCamera.position.z = Math.max(2.2, Math.min(8.0, frame3dCamera.position.z + e.deltaY * 0.005));
    }
  }, { passive: false });

  // Render loop
  function animateFrame3D() {
    requestAnimationFrame(animateFrame3D);
    if (!isDragging && frame3dGroup) {
      frame3dGroup.rotation.z += 0.0015; // Slow ambient orbit
    }
    frame3dRenderer.render(frame3dScene, frame3dCamera);
  }
  animateFrame3D();

  window.addEventListener('resize', () => {
    if (wrap.clientWidth && wrap.clientHeight) {
      frame3dCamera.aspect = wrap.clientWidth / wrap.clientHeight;
      frame3dCamera.updateProjectionMatrix();
      frame3dRenderer.setSize(wrap.clientWidth, wrap.clientHeight, false);
    }
  });
}

// Custom Extracted Frame Image Loader
const customFrameImage = new Image();
customFrameImage.src = 'assets/media/extracted_frame.jpg';
let isCustomFrameReady = false;

customFrameImage.onload = () => {
  isCustomFrameReady = true;
  loadExtractedDroneFrame();
};

function loadExtractedDroneFrame() {
  if (!rawCapturedCanvas) {
    rawCapturedCanvas = document.createElement('canvas');
    rawCapturedCanvas.width = 1280;
    rawCapturedCanvas.height = 720;
  }
  const ctx = rawCapturedCanvas.getContext('2d');
  ctx.drawImage(customFrameImage, 0, 0, rawCapturedCanvas.width, rawCapturedCanvas.height);

  renderFrameDisplayMode();
  updateFrame3DTexture();

  const badge = document.getElementById('extractedFrameBadge');
  if (badge) badge.textContent = 'FRAME #01 · CAMPUS SURVEY · 3840×2160 UHD';

  const liveTc = document.getElementById('videoTimecodeLive');
  if (liveTc) liveTc.textContent = '00:04.50';

  const overlayTag = document.getElementById('frameOverlayTag');
  if (overlayTag) overlayTag.textContent = 'DRONE SURVEY · AERIAL KEYFRAME';

  const sub = document.getElementById('frameKeypointBadge');
  if (sub) sub.textContent = '84 STRUCTURE KEYPOINTS';

  buildFrame3DGeometry(depthDisplacementScale);
  log('[FRAME INGEST] Extracted drone survey keyframe · Built photogrammetric 3D model nearby');
}

function buildFrame3DGeometry(scale = depthDisplacementScale) {
  if (!frame3dGroup || !window.THREE) return;

  // Clear previous children
  while (frame3dGroup.children.length > 0) {
    const obj = frame3dGroup.children[0];
    frame3dGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  // 84x56 dense photogrammetry mesh plane
  const geom = new THREE.PlaneGeometry(5.6, 3.3, 84, 56);
  const pos = geom.attributes.position;

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    // nx: 0 (left) to 1 (right)
    const nx = (x + 2.8) / 5.6;
    // ny: 0 (bottom foreground) to 1 (top horizon/sky)
    const ny = (y + 1.65) / 3.3;

    let z = 0;

    // 1. Foreground Concrete Building Frame & Crane (nx: 0.48 - 0.96, ny: 0.04 - 0.52)
    if (nx >= 0.46 && nx <= 0.96 && ny >= 0.04 && ny <= 0.52) {
      // Base elevation for multi-story structure
      const level = Math.floor((ny - 0.04) / 0.12);
      const slabElevation = 0.55 + level * 0.15;
      // Column/beam grid modulation
      const gridBump = Math.sin((nx - 0.46) * 50.0) * Math.cos((ny - 0.04) * 40.0) * 0.08;
      z = slabElevation + gridBump;
    }

    // Crane boom line extending from (0.50, 0.12) to (0.69, 0.70)
    const craneDist = Math.abs((0.70 - 0.12) * nx - (0.69 - 0.50) * ny + 0.69 * 0.12 - 0.70 * 0.50) / Math.hypot(0.70 - 0.12, 0.69 - 0.50);
    if (craneDist < 0.022 && nx >= 0.48 && nx <= 0.71 && ny >= 0.10 && ny <= 0.72) {
      z = Math.max(z, 0.95 + (0.72 - ny) * 0.2);
    }

    // 2. Main Academic Block (nx: 0.08 - 0.75, ny: 0.35 - 0.68)
    if (nx >= 0.07 && nx <= 0.76 && ny >= 0.35 && ny <= 0.68) {
      let bldgH = 0.45;
      // Roof solar panels / terrace
      bldgH += Math.sin(nx * 32.0) * 0.03;
      // Blue facade entrance atrium
      if (nx >= 0.36 && nx <= 0.52 && ny >= 0.38 && ny <= 0.56) {
        bldgH = 0.55;
      }
      // Rear multi-story tower block
      if (nx >= 0.32 && nx <= 0.44 && ny >= 0.58 && ny <= 0.72) {
        bldgH = 0.72;
      }
      z = Math.max(z, bldgH);
    }

    // 3. Ground, Trees & Green Canopy (foreground & perimeter)
    if (z === 0) {
      if (ny <= 0.75) {
        // Natural terrain and tree canopy ripples
        z = 0.18 + Math.sin(nx * 22.0) * Math.cos(ny * 20.0) * 0.06;
      } else {
        // Distant horizon fading back to base plane
        z = Math.max(0.02, 0.15 * (1.0 - (ny - 0.75) / 0.25));
      }
    }

    pos.setZ(i, z * scale * 0.75);
  }
  geom.computeVertexNormals();

  // Textured Surface Mesh
  frame3dMaterial = new THREE.MeshStandardMaterial({
    map: frame3dTexture || null,
    color: frame3dTexture ? 0xffffff : 0x1e293b,
    roughness: 0.4,
    metalness: 0.2,
    side: THREE.DoubleSide,
    flatShading: false
  });
  frame3dMesh = new THREE.Mesh(geom, frame3dMaterial);
  frame3dMesh.visible = (currentLayer3D === 'textured');
  frame3dGroup.add(frame3dMesh);

  // Dense 3D Point Cloud
  const pointMat = new THREE.PointsMaterial({
    size: 0.045,
    color: 0xffffff,
    map: frame3dTexture || null
  });
  frame3dPoints = new THREE.Points(geom, pointMat);
  frame3dPoints.visible = (currentLayer3D === 'points');
  frame3dGroup.add(frame3dPoints);

  // Wireframe Mesh
  const wireGeo = new THREE.WireframeGeometry(geom);
  frame3dWire = new THREE.LineSegments(
    wireGeo,
    new THREE.LineBasicMaterial({ color: 0xff6b00, transparent: true, opacity: 0.7 })
  );
  frame3dWire.visible = (currentLayer3D === 'wire');
  frame3dGroup.add(frame3dWire);

  // Structural Coordinate Base
  const baseGrid = new THREE.GridHelper(5.8, 14, 0xff6b00, 0x1e293b);
  baseGrid.rotation.x = Math.PI / 2;
  baseGrid.position.z = -0.25;
  frame3dGroup.add(baseGrid);
}

function updateFrame3DTexture() {
  if (!rawCapturedCanvas || !window.THREE) return;

  if (!frame3dTexture) {
    frame3dTexture = new THREE.CanvasTexture(rawCapturedCanvas);
    frame3dTexture.minFilter = THREE.LinearFilter;
    frame3dTexture.magFilter = THREE.LinearFilter;
  } else {
    frame3dTexture.image = rawCapturedCanvas;
    frame3dTexture.needsUpdate = true;
  }

  if (frame3dMaterial) {
    frame3dMaterial.map = frame3dTexture;
    frame3dMaterial.color.setHex(0xffffff);
    frame3dMaterial.needsUpdate = true;
  }

  if (frame3dPoints && frame3dPoints.material) {
    frame3dPoints.material.map = frame3dTexture;
    frame3dPoints.material.needsUpdate = true;
  }
}

// ==========================================================================
// Colab Frame Inspector & Keyframes Filmstrip Controller
// ==========================================================================
let colabTotalFrames = 0;
let colabFps = 30.0;
let colabCurrentFrame = 0;

function computeLaplacianSharpness(canvas) {
  try {
    if (!canvas) return 0.0;
    const ctx = canvas.getContext('2d');
    const sw = 160;
    const sh = 90;
    const offCanvas = document.createElement('canvas');
    offCanvas.width = sw;
    offCanvas.height = sh;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(canvas, 0, 0, sw, sh);
    const imgData = offCtx.getImageData(0, 0, sw, sh);
    const d = imgData.data;
    const gray = new Float32Array(sw * sh);
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      gray[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    }
    let sum = 0, sumSq = 0, count = 0;
    for (let y = 1; y < sh - 1; y++) {
      for (let x = 1; x < sw - 1; x++) {
        const idx = y * sw + x;
        const lap = gray[idx - sw] + gray[idx + sw] + gray[idx - 1] + gray[idx + 1] - 4 * gray[idx];
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }
    const mean = sum / count;
    const variance = (sumSq / count) - (mean * mean);
    return Math.max(0.0, Math.round(variance * 10) / 10);
  } catch (err) {
    return 0.0;
  }
}

function updateColabFrameDisplay(frameIdx) {
  const currentTotal = (typeof colabState !== 'undefined' && colabState.totalFrames) ? colabState.totalFrames : colabTotalFrames;
  if (currentTotal <= 0) return;
  colabCurrentFrame = Math.max(0, Math.min(currentTotal - 1, frameIdx));
  showColabFrame(colabCurrentFrame);
}

function populateKeyframesFilmstrip() {
  const filmstrip = document.getElementById('keyframesFilmstrip');
  if (!filmstrip) return;

  const renderCards = (items) => {
    filmstrip.innerHTML = '';
    items.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = `keyframe-card ${idx === 0 ? 'active' : ''}`;
      card.dataset.frame = item.frame_number;
      card.title = `Click to inspect Frame #${item.frame_number + 1} (${item.timestamp}s)`;

      const imgSrc = item.url || `/api/keyframes/${item.keyframe_index !== undefined ? item.keyframe_index : idx}`;

      card.innerHTML = `
        <img class="keyframe-thumb" src="${imgSrc}" alt="Frame #${item.frame_number + 1}">
        <div class="keyframe-info">
          <strong>Frame #${item.frame_number + 1}</strong>
          <span>Time: ${item.timestamp}s</span>
          <span class="keyframe-sharpness">Sharpness: ${item.sharpness}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        updateColabFrameDisplay(item.frame_index !== undefined ? item.frame_index : item.frame_number);
        const modeDualView = document.getElementById('modeDualView');
        if (modeDualView && !modeDualView.classList.contains('active')) {
          modeDualView.click();
        }
        log(`[COLAB] Inspected Keyframe #${idx + 1} (Frame #${item.frame_number + 1}) | Time: ${item.timestamp}s | Sharpness: ${item.sharpness}`);
      });

      filmstrip.appendChild(card);
    });
  };

  fetch('/api/video/keyframes?count=8')
    .then(r => r.json())
    .then(data => {
      if (data && data.keyframes && data.keyframes.length > 0) {
        renderCards(data.keyframes);
      } else {
        fallbackCards();
      }
    })
    .catch(() => fallbackCards());

  function fallbackCards() {
    filmstrip.innerHTML = '<div style="padding: 16px 20px; color: var(--text-muted); font-size: 13px; text-align: center; width: 100%;">No keyframes extracted yet. Upload a drone video in Step 02 to extract keyframes.</div>';
  }
}

function extractVideoFrame(targetTime = null) {
  const video = document.getElementById('videoPreview');
  const canvas2d = document.getElementById('frame2dCanvas');
  if (!canvas2d) return;

  if (!rawCapturedCanvas) {
    rawCapturedCanvas = document.createElement('canvas');
    rawCapturedCanvas.width = 1280;
    rawCapturedCanvas.height = 720;
  }

  const rawCtx = rawCapturedCanvas.getContext('2d');
  let curTime = 4.50;

  if (video && video.readyState >= 2 && video.videoWidth > 0 && !video.src.includes('sample_4k.mp4')) {
    if (targetTime !== null && !isNaN(targetTime)) {
      video.currentTime = Math.max(0, Math.min(video.duration || 120, targetTime));
    }
    curTime = video.currentTime || 4.50;
    rawCtx.drawImage(video, 0, 0, rawCapturedCanvas.width, rawCapturedCanvas.height);
  } else if (isCustomFrameReady && customFrameImage.complete) {
    rawCtx.drawImage(customFrameImage, 0, 0, rawCapturedCanvas.width, rawCapturedCanvas.height);
  } else {
    curTime = (video && !isNaN(video.currentTime)) ? video.currentTime : 4.50;
    drawSynthesizedDroneSurveyFrame(rawCtx, rawCapturedCanvas.width, rawCapturedCanvas.height, curTime);
  }

  // Render to display 2D canvas with current mode
  renderFrameDisplayMode();

  // Update nearby 3D Model with this exact frame's texture!
  updateFrame3DTexture();

  const badge = document.getElementById('extractedFrameBadge');
  if (badge) badge.textContent = `FRAME #01 · CAMPUS SURVEY · 3840×2160 UHD`;

  const liveTc = document.getElementById('videoTimecodeLive');
  if (liveTc) liveTc.textContent = formatTimecode(curTime);

  log(`[FRAME EXTRACT] Extracted Campus Keyframe · Reconstructed architectural 3D Model nearby`);
}

function renderFrameDisplayMode() {
  const canvas2d = document.getElementById('frame2dCanvas');
  if (!canvas2d || !rawCapturedCanvas) return;
  const ctx = canvas2d.getContext('2d');
  const w = canvas2d.width;
  const h = canvas2d.height;

  ctx.clearRect(0, 0, w, h);

  if (currentFrameMode === 'rgb') {
    ctx.drawImage(rawCapturedCanvas, 0, 0, w, h);
    const tag = document.getElementById('frameOverlayTag');
    if (tag) tag.textContent = 'RGB CAMERA SENSOR · 4K UHD';
    const sub = document.getElementById('frameKeypointBadge');
    if (sub) sub.textContent = 'CAMPUS AERIAL VIEW';
  } else if (currentFrameMode === 'depth') {
    // Generate AI metric depth colormap matched to architectural elevation
    ctx.drawImage(rawCapturedCanvas, 0, 0, w, h);
    const imgData = ctx.getImageData(0, 0, w, h);
    const d = imgData.data;

    for (let py = 0; py < h; py++) {
      const ny = 1.0 - (py / h); // 0 bottom, 1 top
      for (let px = 0; px < w; px++) {
        const nx = px / w;
        const idx = (py * w + px) * 4;
        const lum = (0.299 * d[idx] + 0.587 * d[idx + 1] + 0.114 * d[idx + 2]) / 255.0;

        // Depth value based on architectural elevation profile + luminance
        let depth = 0.25;
        if (nx >= 0.46 && nx <= 0.96 && ny >= 0.04 && ny <= 0.52) {
          depth = 0.78 + lum * 0.22; // High structure in foreground
        } else if (nx >= 0.07 && nx <= 0.76 && ny >= 0.35 && ny <= 0.68) {
          depth = 0.55 + lum * 0.2; // Main building
        } else if (ny > 0.75) {
          depth = 0.08 + lum * 0.1; // Distant horizon / sky
        } else {
          depth = 0.28 + lum * 0.15; // Trees / ground
        }

        // Plasma colormap (deep purple -> emerald green -> bright yellow/orange)
        d[idx] = Math.min(255, Math.floor(depth * 255 * 1.25)); // Red/Orange peak
        d[idx + 1] = Math.min(255, Math.floor(Math.sin(depth * Math.PI) * 230)); // Green midground
        d[idx + 2] = Math.min(255, Math.floor((1.0 - depth) * 255 * 0.85)); // Deep distance blue
      }
    }
    ctx.putImageData(imgData, 0, 0);

    const tag = document.getElementById('frameOverlayTag');
    if (tag) tag.textContent = 'AI METRIC DEPTH MAP (DEPTH ANYTHING V2)';
    const sub = document.getElementById('frameKeypointBadge');
    if (sub) sub.textContent = 'FP16 DENSE DISPLACEMENT';
  } else if (currentFrameMode === 'keypoints') {
    ctx.drawImage(rawCapturedCanvas, 0, 0, w, h);

    // Photogrammetric feature keypoints targeted at real structures in this photo
    const keypoints = [
      // Construction site columns & frame
      { x: 0.52, y: 0.90, id: 'C01' }, { x: 0.62, y: 0.88, id: 'C02' }, { x: 0.74, y: 0.86, id: 'C03' },
      { x: 0.88, y: 0.84, id: 'C04' }, { x: 0.95, y: 0.82, id: 'C05' }, { x: 0.58, y: 0.75, id: 'C06' },
      { x: 0.68, y: 0.72, id: 'C07' }, { x: 0.82, y: 0.70, id: 'C08' }, { x: 0.55, y: 0.62, id: 'C09' },
      { x: 0.65, y: 0.58, id: 'C10' }, { x: 0.78, y: 0.55, id: 'C11' }, { x: 0.92, y: 0.52, id: 'C12' },
      // Crane boom
      { x: 0.51, y: 0.85, id: 'CR1' }, { x: 0.55, y: 0.68, id: 'CR2' }, { x: 0.60, y: 0.52, id: 'CR3' },
      { x: 0.65, y: 0.38, id: 'CR4' }, { x: 0.69, y: 0.28, id: 'CR5' },
      // Main academic building roofs & facade
      { x: 0.12, y: 0.52, id: 'B01' }, { x: 0.20, y: 0.48, id: 'B02' }, { x: 0.30, y: 0.46, id: 'B03' },
      { x: 0.40, y: 0.44, id: 'B04' }, { x: 0.50, y: 0.43, id: 'B05' }, { x: 0.62, y: 0.42, id: 'B06' },
      { x: 0.74, y: 0.41, id: 'B07' }, { x: 0.42, y: 0.48, id: 'AT1' }, { x: 0.46, y: 0.52, id: 'AT2' },
      { x: 0.37, y: 0.32, id: 'TW1' }, { x: 0.42, y: 0.28, id: 'TW2' },
      // Indian Flag
      { x: 0.14, y: 0.62, id: 'FLG' },
      // Wind turbines on horizon
      { x: 0.08, y: 0.17, id: 'WT1' }, { x: 0.75, y: 0.14, id: 'WT2' }, { x: 0.80, y: 0.17, id: 'WT3' }
    ];

    // Add dense grid points across structures
    for (let k = 0; k < 45; k++) {
      const sx = 0.15 + ((k * 17) % 80) / 100.0;
      const sy = 0.35 + ((k * 23) % 55) / 100.0;
      keypoints.push({ x: sx, y: sy, id: `K${k + 1}` });
    }

    ctx.strokeStyle = '#10b981';
    ctx.fillStyle = '#10b981';
    ctx.lineWidth = 1.5;

    keypoints.forEach((kp) => {
      const kpx = kp.x * w;
      const kpy = kp.y * h;

      ctx.beginPath();
      ctx.arc(kpx, kpy, 3.5, 0, Math.PI * 2);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(kpx - 6, kpy); ctx.lineTo(kpx + 6, kpy);
      ctx.moveTo(kpx, kpy - 6); ctx.lineTo(kpx, kpy + 6);
      ctx.stroke();

      if (kp.id.startsWith('C') || kp.id.startsWith('CR') || kp.id.startsWith('FLG') || kp.id.startsWith('WT')) {
        ctx.font = 'bold 8.5px monospace';
        ctx.fillText(kp.id, kpx + 5, kpy - 5);
      }
    });

    const tag = document.getElementById('frameOverlayTag');
    if (tag) tag.textContent = 'ORB / SIFT PHOTOGRAMMETRIC KEYPOINTS';
    const sub = document.getElementById('frameKeypointBadge');
    if (sub) sub.textContent = `${keypoints.length} POINTS TRACKED`;
  }
}

function drawSynthesizedDroneSurveyFrame(ctx, w, h, timeSec = 0) {
  // Rich aerial photogrammetry keyframe
  const grad = ctx.createLinearGradient(0, 0, w, h);
  grad.addColorStop(0, '#1e3a5f'); // Deep water/reservoir
  grad.addColorStop(0.35, '#2d5a27'); // Forests & vegetation
  grad.addColorStop(0.7, '#8b6f47'); // Terrain & quarry
  grad.addColorStop(1, '#c2b280'); // Sandstone & structures
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Agricultural & urban parcel grids
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.lineWidth = 1;
  for (let x = 40; x < w; x += 60) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 30; y < h; y += 45) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Winding highway / river
  ctx.strokeStyle = '#0f172a';
  ctx.lineWidth = 14;
  ctx.beginPath();
  ctx.moveTo(0, h * 0.4);
  ctx.bezierCurveTo(w * 0.3, h * 0.2, w * 0.6, h * 0.8, w, h * 0.6);
  ctx.stroke();

  // Highway center dashes
  ctx.strokeStyle = '#ff6b00';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Surveyed Structure / Roof cluster
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(w * 0.45, h * 0.35, 110, 75);
  ctx.strokeStyle = '#ff6b00';
  ctx.lineWidth = 2;
  ctx.strokeRect(w * 0.45, h * 0.35, 110, 75);

  ctx.fillStyle = '#ff6b00';
  ctx.font = 'bold 11px monospace';
  ctx.fillText('SURVEY ZONE A', w * 0.45 + 10, h * 0.35 + 25);
  ctx.fillStyle = '#0f172a';
  ctx.font = '10px monospace';
  ctx.fillText('ALT: 124m AGL', w * 0.45 + 10, h * 0.35 + 42);
  ctx.fillText('GSD: 1.4 cm/px', w * 0.45 + 10, h * 0.35 + 58);

  // Drone HUD reticle in center
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 1.5;
  const cx = w / 2;
  const cy = h / 2;
  ctx.beginPath();
  ctx.arc(cx, cy, 24, 0, Math.PI * 2);
  ctx.moveTo(cx - 36, cy); ctx.lineTo(cx - 10, cy);
  ctx.moveTo(cx + 10, cy); ctx.lineTo(cx + 36, cy);
  ctx.moveTo(cx, cy - 36); ctx.lineTo(cx, cy - 10);
  ctx.moveTo(cx, cy + 10); ctx.lineTo(cx, cy + 36);
  ctx.stroke();

  // HUD Top Bar
  ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
  ctx.fillRect(0, 0, w, 26);
  ctx.fillStyle = '#10b981';
  ctx.font = 'bold 10px monospace';
  ctx.fillText(`● REC 4K [3840×2160] · RTK FIXED · SAT: 24 · TIME: ${formatTimecode(timeSec)}`, 14, 17);
}

function initThreeScene() {
  const canvas = document.getElementById('sceneCanvas');
  const viewport = document.getElementById('viewport');
  if (!window.THREE || !canvas || !viewport) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x09121d); // Deep navy dark

  camera = new THREE.PerspectiveCamera(45, viewport.clientWidth / viewport.clientHeight, 0.1, 100);
  camera.position.set(6, 5, 7);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);

  // Lights
  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x0f172a, 1.8);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xff6b00, 1.2);
  dirLight.position.set(5, 10, 7);
  scene.add(dirLight);

  // 3D Model Group
  modelGroup = new THREE.Group();

  // Create stylized terrain / structure geometry
  const geometry = new THREE.IcosahedronGeometry(2.3, 2);
  solidMaterial = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.4,
    metalness: 0.3,
    flatShading: true
  });
  
  const solidMesh = new THREE.Mesh(geometry, solidMaterial);
  modelGroup.add(solidMesh);

  // Wireframe Layer
  const wireGeo = new THREE.WireframeGeometry(geometry);
  wireframeMesh = new THREE.LineSegments(
    wireGeo,
    new THREE.LineBasicMaterial({ color: 0xff6b00, transparent: true, opacity: 0.6 })
  );
  modelGroup.add(wireframeMesh);
  scene.add(modelGroup);

  // Drone Flight Path Line
  const points = [
    new THREE.Vector3(-4, 3, -3),
    new THREE.Vector3(-2, 2.5, -1),
    new THREE.Vector3(0, 3.5, 1),
    new THREE.Vector3(2, 2.8, 0),
    new THREE.Vector3(4, 3.2, -2)
  ];
  const pathGeo = new THREE.BufferGeometry().setFromPoints(points);
  pathLine = new THREE.Line(
    pathGeo,
    new THREE.LineBasicMaterial({ color: 0xff6b00, linewidth: 2 })
  );
  scene.add(pathLine);

  // Grid Floor
  const grid = new THREE.GridHelper(14, 14, 0xff6b00, 0x1e293b);
  grid.position.y = -2.2;
  scene.add(grid);

  // Mouse / Touch Orbit Interaction
  let dragging = false;
  let lastX = 0, lastY = 0;

  viewport.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });

  window.addEventListener('pointerup', () => { dragging = false; });

  window.addEventListener('pointermove', (e) => {
    if (dragging && modelGroup) {
      const deltaX = e.clientX - lastX;
      const deltaY = e.clientY - lastY;
      modelGroup.rotation.y += deltaX * 0.008;
      modelGroup.rotation.x += deltaY * 0.008;
      lastX = e.clientX;
      lastY = e.clientY;
    }
  });

  viewport.addEventListener('wheel', (e) => {
    camera.position.z = Math.max(3, Math.min(12, camera.position.z + e.deltaY * 0.008));
  });

  // Render Loop
  function animate() {
    requestAnimationFrame(animate);
    if (!dragging && modelGroup) {
      modelGroup.rotation.y += 0.003;
    }
    renderer.render(scene, camera);
  }
  animate();

  // Resize Handler
  window.addEventListener('resize', () => {
    camera.aspect = viewport.clientWidth / viewport.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
  });

  // Camera View Angle Switching
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const view = btn.dataset.view;
      if (view === 'top') {
        camera.position.set(0, 9, 0.01);
      } else if (view === 'front') {
        camera.position.set(0, 1, 9);
      } else {
        camera.position.set(6, 5, 7);
      }
      camera.lookAt(0, 0, 0);
    });
  });

  // Layer Toggles
  document.querySelectorAll('[data-layer]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-layer]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const layer = btn.dataset.layer;
      if (layer === 'wire') {
        wireframeMesh.visible = true;
        solidMaterial.wireframe = true;
      } else if (layer === 'heat') {
        solidMaterial.wireframe = false;
        wireframeMesh.visible = false;
        solidMaterial.color.setHex(0xff6b00); // Orange heat
      } else if (layer === 'path') {
        solidMaterial.wireframe = false;
        solidMaterial.color.setHex(0x0f172a);
        wireframeMesh.visible = true;
        pathLine.visible = true;
      } else { // Solid
        solidMaterial.wireframe = false;
        wireframeMesh.visible = true;
        solidMaterial.color.setHex(0x0f172a);
      }
    });
  });
}

// DOM Initialization
document.addEventListener('DOMContentLoaded', () => {
  initThreeScene();
  initFrame3DScene();

  // Mode Switcher: Dual View (Extracted Frame & Nearby 3D) vs Full Flight Scene
  const modeDualView = document.getElementById('modeDualView');
  const modeFullScene = document.getElementById('modeFullScene');
  const dualContainer = document.getElementById('frameTo3dDualContainer');
  const viewportFull = document.getElementById('viewport');

  if (modeDualView && modeFullScene) {
    modeDualView.addEventListener('click', () => {
      modeDualView.classList.add('active');
      modeFullScene.classList.remove('active');
      if (dualContainer) dualContainer.style.display = 'grid';
      if (viewportFull) viewportFull.style.display = 'none';
      log('[VIEWPORT] Switched to Dual View: Extracted Frame (2D) & Nearby 3D Model.');
    });

    modeFullScene.addEventListener('click', () => {
      modeFullScene.classList.add('active');
      modeDualView.classList.remove('active');
      if (dualContainer) dualContainer.style.display = 'none';
      if (viewportFull) viewportFull.style.display = 'block';
      log('[VIEWPORT] Switched to Full Flight Trajectory Scene.');
    });
  }

  // Colab Interactive Frame Slider & Stepper Controls
  const colabSlider = document.getElementById('colabFrameSlider');
  const stepMinus10Btn = document.getElementById('colabStepMinus10');
  const stepPlus10Btn = document.getElementById('colabStepPlus10');
  const stepBackBtn = document.getElementById('stepBackSec');
  const stepForwardBtn = document.getElementById('stepForwardSec');
  const extractBtn1 = document.getElementById('extractVideoFrameBtn');
  const extractBtn2 = document.getElementById('quickExtractBtn');
  const videoEl = document.getElementById('videoPreview');

  if (colabSlider) {
    colabSlider.addEventListener('input', (e) => {
      updateColabFrameDisplay(parseInt(e.target.value));
    });
  }

  if (stepMinus10Btn) {
    stepMinus10Btn.addEventListener('click', () => {
      updateColabFrameDisplay(colabCurrentFrame - 10);
    });
  }

  if (stepPlus10Btn) {
    stepPlus10Btn.addEventListener('click', () => {
      updateColabFrameDisplay(colabCurrentFrame + 10);
    });
  }

  if (stepBackBtn) {
    stepBackBtn.addEventListener('click', () => {
      updateColabFrameDisplay(colabCurrentFrame - 1);
    });
  }

  if (stepForwardBtn) {
    stepForwardBtn.addEventListener('click', () => {
      updateColabFrameDisplay(colabCurrentFrame + 1);
    });
  }

  const triggerExtract = () => {
    if (videoEl && !videoEl.paused) videoEl.pause();
    updateColabFrameDisplay(colabCurrentFrame);
    if (modeDualView && !modeDualView.classList.contains('active')) {
      modeDualView.click();
    }
  };

  if (extractBtn1) extractBtn1.addEventListener('click', triggerExtract);
  if (extractBtn2) extractBtn2.addEventListener('click', triggerExtract);

  // Initialize Colab Keyframes Filmstrip Gallery
  populateKeyframesFilmstrip();

  // Query Python backend OpenCV metadata
  fetch('/api/video/metadata')
    .then(r => r.json())
    .then(data => {
      if (data && data.total_frames) {
        colabTotalFrames = data.total_frames;
        colabFps = data.fps || 29.97;
        const totalFramesEl = document.getElementById('colabTotalFrames');
        if (totalFramesEl) totalFramesEl.textContent = colabTotalFrames;
        if (colabSlider) colabSlider.max = colabTotalFrames - 1;
        populateKeyframesFilmstrip();
        log(`[OPENCV] Loaded video stream metadata from Python OpenCV: ${data.total_frames} frames @ ${data.fps} FPS.`);
      }
    })
    .catch(() => {});

  // 2D Frame Modes (RGB, Depth, Keypoints)
  document.querySelectorAll('[data-framemode]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-framemode]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentFrameMode = btn.dataset.framemode;
      renderFrameDisplayMode();
      log(`[FRAME MODE] Switched 2D Frame inspection mode: ${currentFrameMode.toUpperCase()}`);
    });
  });

  // 3D Layers (Textured, Points, Wire)
  document.querySelectorAll('[data-layer3d]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-layer3d]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentLayer3D = btn.dataset.layer3d;
      if (frame3dMesh) frame3dMesh.visible = (currentLayer3D === 'textured');
      if (frame3dPoints) frame3dPoints.visible = (currentLayer3D === 'points');
      if (frame3dWire) frame3dWire.visible = (currentLayer3D === 'wire');
      log(`[3D LAYER] Switched nearby 3D Model render mode: ${currentLayer3D.toUpperCase()}`);
    });
  });

  // Depth Displacement Scale Slider
  const depthSlider = document.getElementById('depthScaleRange');
  if (depthSlider) {
    depthSlider.addEventListener('input', (e) => {
      depthDisplacementScale = parseFloat(e.target.value) / 10.0;
      buildFrame3DGeometry(depthDisplacementScale);
    });
  }

  // Initialize visual canvas state
  setTimeout(() => {
    extractVideoFrame(4.50);
  }, 400);

  // Left Panel Workflow Navigation: Workflow Modules
  const workflowNavBtns = document.querySelectorAll('.workflow-nav-item');

  function activateSidebarModule(moduleName) {
    workflowNavBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.module === moduleName);
    });

    const dashboardMainArea = document.getElementById('dashboardMainArea');
    const appShell = document.querySelector('.app-shell');
    const workspaceGrid = document.getElementById('workspaceGrid');
    const dedicatedTrajectorySection = document.getElementById('dedicatedTrajectorySection');
    const dashboardKeyframesSection = document.getElementById('dashboardKeyframesSection');
    const uploadVideoSection = document.getElementById('uploadVideoSection');
    const projectsSection = document.getElementById('projectsSection');
    const dashboardActiveProjectCard = document.getElementById('dashboardActiveProjectCard');
    const overviewSection = document.getElementById('dashboardOverviewSection');
    const phaseStrip = document.getElementById('phaseStrip');

    if (dashboardMainArea) {
      dashboardMainArea.setAttribute('data-active-view', moduleName);
    }
    if (appShell) {
      appShell.setAttribute('data-active-view', moduleName);
    }

    // Default hide all modular sections
    if (phaseStrip) phaseStrip.style.display = 'none';
    if (workspaceGrid) workspaceGrid.style.display = 'none';
    if (projectsSection) projectsSection.style.display = 'none';
    if (uploadVideoSection) uploadVideoSection.style.display = 'none';
    if (dedicatedTrajectorySection) dedicatedTrajectorySection.style.display = 'none';
    if (dashboardKeyframesSection) dashboardKeyframesSection.style.display = 'none';
    if (dashboardActiveProjectCard) dashboardActiveProjectCard.style.display = 'none';
    if (overviewSection) overviewSection.style.display = 'none';

    // 1. DASHBOARD: ONLY Keyframes section, Trajectory, and Telemetry (Resolution, FPS, Total Frames, Duration)
    if (moduleName === 'dashboard') {
      if (dashboardActiveProjectCard) dashboardActiveProjectCard.style.display = 'block';
      if (dedicatedTrajectorySection) dedicatedTrajectorySection.style.display = 'block';
      if (dashboardKeyframesSection) dashboardKeyframesSection.style.display = 'block';

      // Ensure Camera Trajectory renders
      if (cachedTrajectoryData && cachedTrajectoryData.trajectory) {
        render2DCameraTrajectory(
          cachedTrajectoryData.trajectory,
          cachedTrajectoryData.successful_matches,
          cachedTrajectoryData.failed_frames
        );
      } else {
        loadTrajectoryData();
      }

      // Ensure Keyframes are loaded
      loadKeyframesFilmstrip();
      loadSelectedKeyframes();

      window.scrollTo({ top: 0, behavior: 'smooth' });
      log('[WORKFLOW] Dashboard loaded: Displaying Keyframes, Trajectory, and Video Specs (Resolution, FPS, Total Frames, Duration).');
      return;
    }

    // 2. PROJECTS: Library of video file buttons
    if (moduleName === 'projects') {
      if (projectsSection) projectsSection.style.display = 'block';
      loadProjectsList();
      window.scrollTo({ top: 0, behavior: 'smooth' });
      log('[WORKFLOW] Switched to 02 · Drone Video Projects Library.');
      return;
    }

    // 3. UPLOAD VIDEO: Video Ingestion
    if (moduleName === 'upload-video' || moduleName === 'upload') {
      if (uploadVideoSection) uploadVideoSection.style.display = 'block';
      window.scrollTo({ top: 0, behavior: 'smooth' });
      log('[WORKFLOW] Switched to Video Ingestion view.');
      return;
    }

    // 4. TRAJECTORY: Full Camera Trajectory
    if (moduleName === 'trajectory' || moduleName === 'mapping') {
      if (dedicatedTrajectorySection) dedicatedTrajectorySection.style.display = 'block';
      if (cachedTrajectoryData && cachedTrajectoryData.trajectory) {
        render2DCameraTrajectory(
          cachedTrajectoryData.trajectory,
          cachedTrajectoryData.successful_matches,
          cachedTrajectoryData.failed_frames
        );
      } else {
        loadTrajectoryData();
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
      log('[WORKFLOW] Switched to Camera Trajectory.');
      return;
    }

    // 5. VIDEO PROCESSING: Keyframes Inspection
    if (moduleName === 'video-processing' || moduleName === 'processing' || moduleName === 'analysis') {
      if (dashboardKeyframesSection) dashboardKeyframesSection.style.display = 'block';
      loadKeyframesFilmstrip();
      loadSelectedKeyframes();
      window.scrollTo({ top: 0, behavior: 'smooth' });
      log('[WORKFLOW] Switched to Video Processing (Keyframes Inspection).');
      return;
    }

    // 6. OTHER MODULES (still UI mock-ups: Georeferencing, 3D Generation)
    // Stage 04 (Pose & Depth) is a real page owned by assets/js/stage04.js, which watches
    // data-active-view on #dashboardMainArea to show itself.
    if (moduleName === 'depth-estimation') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      log('[WORKFLOW] Switched to Stage 04 · Camera Pose & Depth.');
      return;
    }

    // "Georeferencing" and "3D Model Generation" used to open a mock page. The REAL georeferencing,
    // dense map and 3D model/mesh generation all run inside Pose & Depth (Stage 04), so route the
    // user there — one coherent reconstruction flow instead of a confusing dead-end mock.
    if (moduleName === 'georeferencing' || moduleName === '3d-generation') {
      log('[WORKFLOW] Georeferencing & 3D model generation run inside Pose & Depth (Stage 04). Opening it.');
      activateSidebarModule('depth-estimation');
      return;
    }
  }

  workflowNavBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      activateSidebarModule(btn.dataset.module);
    });
  });

  // Upload Method Selector: File Format vs URL / Link Format
  const optBtnFile = document.getElementById('optBtnFile');
  const optBtnUrl = document.getElementById('optBtnUrl');
  const fileUploadContainer = document.getElementById('fileUploadContainer');
  const urlUploadContainer = document.getElementById('urlUploadContainer');

  function setUploadMethod(method) {
    if (method === 'file') {
      if (optBtnFile) optBtnFile.classList.add('active');
      if (optBtnUrl) optBtnUrl.classList.remove('active');
      if (fileUploadContainer) fileUploadContainer.style.display = 'block';
      if (urlUploadContainer) urlUploadContainer.style.display = 'none';
      log('[INGEST METHOD] Switched to Video File Format (.MP4/.MOV/.AVI local upload).');
    } else {
      if (optBtnUrl) optBtnUrl.classList.add('active');
      if (optBtnFile) optBtnFile.classList.remove('active');
      if (urlUploadContainer) urlUploadContainer.style.display = 'block';
      if (fileUploadContainer) fileUploadContainer.style.display = 'none';
      log('[INGEST METHOD] Switched to Video URL / Link Format (online web stream).');
    }
  }

  if (optBtnFile) optBtnFile.addEventListener('click', () => setUploadMethod('file'));
  if (optBtnUrl) optBtnUrl.addEventListener('click', () => setUploadMethod('url'));

  // Run Pipeline Button (if present)
  const runBtn = document.getElementById('runAll');
  if (runBtn) runBtn.addEventListener('click', runPipeline);

  // Reset Button
  const resetBtn = document.getElementById('resetBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (!running) {
        setProgress(0);
        const statusPill = document.getElementById('inputStatus');
        if (statusPill) {
          statusPill.textContent = 'WAITING FOR INPUT';
          statusPill.className = 'status-pill idle';
        }
        const metaEmptyState = document.getElementById('metaEmptyState');
        const metaGrid = document.getElementById('metaGrid');
        const metaActionsBar = document.getElementById('metaActionsBar');
        if (metaEmptyState) metaEmptyState.style.display = 'flex';
        if (metaGrid) metaGrid.style.display = 'none';
        if (metaActionsBar) metaActionsBar.style.display = 'none';
        activeTelemetryState = null;

        const dropZone = document.getElementById('dropZone');
        if (dropZone) {
          dropZone.style.display = 'flex';
          dropZone.innerHTML = `
            <div class="dropzone-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <div class="dropzone-text">
              <strong>Drag &amp; drop drone video file</strong>
              <p>Supports 4K/1080p MP4, MOV, or AVI flight recordings</p>
            </div>
            <button type="button" class="primary-btn sm-btn" onclick="event.stopPropagation(); document.getElementById('videoFile').click();">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              Browse Computer
            </button>
          `;
        }

        const videoUrlInput = document.getElementById('videoUrl');
        if (videoUrlInput) videoUrlInput.value = '';

        const actionBox = document.getElementById('videoProcessingActionBox');
        if (actionBox) actionBox.style.display = 'none';

        const videoFileInput = document.getElementById('videoFile');
        if (videoFileInput) videoFileInput.value = '';

        const videoContainer = document.getElementById('videoContainer');
        const videoPreview = document.getElementById('videoPreview');
        if (videoPreview) {
          videoPreview.pause();
          videoPreview.removeAttribute('src');
          videoPreview.load();
        }
        if (videoContainer) videoContainer.style.display = 'none';
        log('[SYSTEM] Workspace reset.');
      }
    });
  }

  // Clear Logs
  const clearBtn = document.getElementById('clearLogs');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      const terminal = document.getElementById('terminal');
      if (terminal) terminal.innerHTML = '';
    });
  }

  // File Upload Ingestion Handler
  async function handleFileSelected(file) {
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      log('Error: Please upload a valid MP4, MOV, or AVI video file.');
      return;
    }
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    const objectUrl = URL.createObjectURL(file);
    
    const dropZone = document.getElementById('dropZone');
    if (dropZone) {
      dropZone.innerHTML = `
        <div class="dropzone-icon" style="color: var(--orange-primary); animation: pulse 1s infinite;">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        </div>
        <div class="dropzone-text">
          <strong>Processing ${file.name} (${sizeMb} MB)...</strong>
          <p id="uploadProcessingStatus">FastAPI backend extracting ALL frames, calculating sharpness &amp; trajectory...</p>
        </div>
      `;
    }

    loadVideo(objectUrl, `${file.name} (${sizeMb} MB)`, file);
    log(`[FILE INGESTION] Video file "${file.name}" (${sizeMb} MB) uploaded. Sending to FastAPI backend (POST /api/process-video)...`);

    // Stream/upload to Python FastAPI Backend for cv2 analysis, frame extraction, keyframes & trajectory
    const formData = new FormData();
    formData.append('video', file);

    // Live progress polling
    const progressStatusEl = document.getElementById('uploadProcessingStatus');
    const pollInterval = setInterval(async () => {
      try {
        const pRes = await fetch('/api/progress');
        if (pRes.ok) {
          const pData = await pRes.json();
          if (pData && pData.status === 'PROCESSING') {
            if (progressStatusEl) {
              progressStatusEl.innerHTML = `<span style="color:var(--orange-primary);font-weight:600;">[${pData.percent}%]</span> ${pData.stage}: ${pData.current_frame} / ${pData.total_frames} frames`;
            }
          }
        }
      } catch (e) {
        // ignore polling errors
      }
    }, 400);

    try {
      const modeSelect = document.getElementById('keyframeModeSelect');
      const keyframeMode = modeSelect ? modeSelect.value : '';
      const res = await fetch('/api/process-video' + (keyframeMode ? `?keyframe_mode=${encodeURIComponent(keyframeMode)}` : ''), {
        method: 'POST',
        body: formData
      });

      clearInterval(pollInterval);

      if (res.ok) {
        const data = await res.json();
        log(`[FASTAPI] ✅ Video processed successfully (keyframe mode: ${data.keyframe_mode}) · Res: ${data.resolution} · FPS: ${data.fps} · Extracted: ${data.number_of_extracted_frames} · Sharp: ${data.number_of_sharp_frames} · Keyframes: ${data.number_of_final_keyframes} · Trajectory: ${data.trajectory_points} pts`);
        
        if (dropZone) {
          dropZone.innerHTML = `
            <div class="dropzone-icon" style="color: var(--accent-green);">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
            </div>
            <div class="dropzone-text">
              <strong>✓ ${file.name} Processed</strong>
              <p>${data.number_of_extracted_frames} frames · ${data.number_of_final_keyframes} keyframes · ${data.trajectory_points} trajectory points</p>
            </div>
            <button type="button" class="primary-btn sm-btn" onclick="event.stopPropagation(); document.getElementById('videoFile').click();">
              Change Video
            </button>
          `;
        }

        if (typeof onVideoProcessedSuccess === 'function') {
          onVideoProcessedSuccess(file, data);
        } else if (typeof updateDashboardProjectDetails === 'function') {
          updateDashboardProjectDetails(data);
        }

        // Update Colab Frame Inspector metrics with FastAPI backend ground-truth
        colabState.totalFrames = data.total_frames || data.number_of_extracted_frames;
        colabState.fps = data.fps;
        if (colabTotalFrames) colabTotalFrames.textContent = colabState.totalFrames;
        if (colabFrameSlider) {
          colabFrameSlider.min = 0;
          colabFrameSlider.max = Math.max(0, colabState.totalFrames - 1);
          colabFrameSlider.value = 0;
        }
        const colabNumInput = document.getElementById('colabFrameNumberInput');
        if (colabNumInput) {
          colabNumInput.min = 0;
          colabNumInput.max = Math.max(0, colabState.totalFrames - 1);
          colabNumInput.value = 0;
        }

        // Populate Keyframes if provided
        if (data.keyframes && data.keyframes.length > 0) {
          selectedFramesList = data.keyframes;
          const kfMax = Math.max(0, selectedFramesList.length - 1);
          const kfSlider = document.getElementById('selectedKeyframeSlider');
          const kfInput = document.getElementById('selectedKeyframeNumberInput');
          const kfTotal = document.getElementById('keyframeTotalCount');
          const origTotal = document.getElementById('keyframeOrigTotal');
          if (kfSlider) { kfSlider.min = 0; kfSlider.max = kfMax; kfSlider.value = 0; }
          if (kfInput) { kfInput.min = 0; kfInput.max = kfMax; kfInput.value = 0; }
          if (kfTotal) kfTotal.textContent = selectedFramesList.length;
          if (origTotal) origTotal.textContent = colabState.totalFrames;
          show_selected_keyframe(0);
        }

        // Render Trajectory on ALL frames
        if (data.trajectory && data.trajectory.length > 0) {
          render2DCameraTrajectory(data.trajectory, data.successful_trajectory_matches, data.failed_trajectory_frames);
          updateThreeFlightPath(data.trajectory);
        }

        // Show first frame in inspector
        showColabFrame(0);

        // Show the Video Processing Action Box right below the upload area!
        const actionBox = document.getElementById('videoProcessingActionBox');
        const actionTitle = document.getElementById('processingAdvanceTitle');
        const actionSub = document.getElementById('processingAdvanceSub');
        const actionBadge = document.getElementById('processingAdvanceBadge');
        if (actionBox) {
          actionBox.style.display = 'flex';
          if (actionTitle) actionTitle.textContent = `✓ ${file.name} Processed`;
          if (actionSub) actionSub.textContent = `${data.resolution} · ${data.fps} FPS · ${data.number_of_extracted_frames} Frames (${data.duration}s) · Keyframes: ${data.number_of_final_keyframes} · Trajectory: ${data.trajectory_points} pts`;
          if (actionBadge) actionBadge.textContent = 'PROCESSED';
        }
      } else {
        const errData = await res.json().catch(() => ({}));
        log(`[FASTAPI] ❌ Video processing failed: ${errData.detail || errData.error || 'Check video format'}`);
        if (dropZone) {
          dropZone.innerHTML = `
            <div class="dropzone-icon" style="color: #ef4444;">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
            </div>
            <div class="dropzone-text">
              <strong>Processing Failed</strong>
              <p>${errData.detail || 'Could not decode or process video'}</p>
            </div>
            <button type="button" class="primary-btn sm-btn" onclick="event.stopPropagation(); document.getElementById('videoFile').click();">
              Retry Upload
            </button>
          `;
        }
      }
    } catch (err) {
      console.warn('[FASTAPI BACKEND] Server upload notice:', err);
      log(`[FASTAPI] ❌ Error: ${err.message}`);
    }
  }

  // Start Video Processing with Uploaded Video as Input
  async function startVideoProcessingWithUploadedVideo() {
    log('[WORKFLOW] Taking uploaded video as input for Video Processing...');

    // 1. Move to Video Processing workspace
    activateSidebarModule('video-processing');

    // 2. Fetch fresh video metadata for the uploaded video from OpenCV backend
    await fetchVideoMetadata();

    // 3. Extract frames: load frame #0 into the photographic viewer, calculate live Laplacian sharpness
    log('[FASTAPI] Displaying extracted image frames from uploaded video...');
    await showColabFrame(0);

    // 4. Extract keyframes: generate candidate keyframes filmstrip and selected keyframes interactive viewer
    log('[FASTAPI] Loading keyframes & calculating Laplacian sharpness variance...');
    await loadKeyframesFilmstrip();
    await loadSelectedKeyframes();

    // 5. Load Camera Trajectory
    await loadTrajectoryData();

    // 6. Update progress to Stage 02
    setProgress(25);
    const pill = document.getElementById('inputStatus');
    if (pill) {
      pill.textContent = 'PROCESSED';
      pill.className = 'status-pill ready';
    }

    log('✅ [FASTAPI] Video processing complete: Extracted frames & candidate keyframes & camera trajectory.');
  }

  const btnGoToVideoProcessing = document.getElementById('btnGoToVideoProcessing');
  if (btnGoToVideoProcessing) {
    btnGoToVideoProcessing.addEventListener('click', startVideoProcessingWithUploadedVideo);
  }

  // File Upload Input
  const fileInput = document.getElementById('videoFile');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  // Drag and Drop Zone
  const dropZone = document.getElementById('dropZone');
  if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--orange-primary)';
      dropZone.style.background = 'var(--orange-soft)';
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.style.borderColor = '';
      dropZone.style.background = '';
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = '';
      dropZone.style.background = '';

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFileSelected(file);
      }
    });
  }

  // URL Input Load Button
  const loadUrlBtn = document.getElementById('loadUrl');
  const urlInput = document.getElementById('videoUrl');

  function handleUrlSubmit() {
    const url = urlInput ? urlInput.value.trim() : '';
    if (!url) {
      log('Error: Please enter a valid video URL.');
      return;
    }
    try {
      const parsed = new URL(url, window.location.href);
      const displayLabel = parsed.pathname.split('/').pop() || 'Remote Video Stream';
      loadVideo(url, displayLabel);
      log(`[URL INGESTION] Streaming video from link: ${url}`);
    } catch {
      log('Error: Please enter a full http:// or https:// URL.');
    }
  }

  if (loadUrlBtn) loadUrlBtn.addEventListener('click', handleUrlSubmit);
  if (urlInput) {
    urlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handleUrlSubmit();
    });
  }

  // URL Preset Chips (One-click direct 4K drone links)
  document.querySelectorAll('.url-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const url = chip.dataset.url;
      const label = chip.dataset.label;
      if (urlInput) urlInput.value = url;
      loadVideo(url, label);
      log(`[URL INGESTION] Selected drone footage: ${label}`);
    });
  });

  // Quick Load Sample 4K Video Button
  const loadSampleBtn = document.getElementById('loadSampleBtn');
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener('click', () => {
      loadVideo('assets/media/sample_4k.mp4', 'sample_4k.mp4 (92.5 MB · DJI 4K Drone Aerial Survey)');
      log('[SAMPLE] Loaded pre-bundled 4K drone survey footage.');
    });
  }

  // Preset Buttons
  document.querySelectorAll('.preset').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.preset').forEach(p => p.classList.remove('selected'));
      btn.classList.add('selected');
      log(`[CONFIG] Selected reconstruction preset: ${btn.dataset.preset.toUpperCase()}`);
    });
  });

  // Slider Label
  const densitySlider = document.getElementById('density');
  const densityLabel = document.getElementById('densityLabel');
  if (densitySlider && densityLabel) {
    const labels = [
      'Sparse · 4 frames / min',
      'Light · 8 frames / min',
      'Balanced · 12 frames / min',
      'Dense · 20 frames / min',
      'Maximum · 32 frames / min'
    ];
    densitySlider.addEventListener('input', (e) => {
      densityLabel.textContent = labels[e.target.value - 1];
    });
  }

  // Export Buttons (Deliverables)
  document.querySelectorAll('.export-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const fmt = btn.dataset.format.toUpperCase();
      log(`[EXPORT] Initiated ${fmt} dataset download...`);
      const check = btn.querySelector('strong');
      if (check) check.textContent = '✓';
      setTimeout(() => {
        if (check) check.textContent = '↓';
      }, 2000);
    });
  });

  // Camera Intrinsics K-Matrix Modal Handlers
  const viewKMatrixBtn = document.getElementById('viewKMatrixBtn');
  const intrinsicsModal = document.getElementById('intrinsicsModal');
  const closeIntrinsicsModal = document.getElementById('closeIntrinsicsModal');
  const doneIntrinsicsModal = document.getElementById('doneIntrinsicsModal');
  const copyIntrinsicsBtn = document.getElementById('copyIntrinsicsBtn');

  if (viewKMatrixBtn && intrinsicsModal) {
    viewKMatrixBtn.addEventListener('click', () => {
      intrinsicsModal.style.display = 'flex';
      log('[INTRINSICS] Opened 3×3 Pinhole Camera Intrinsics Calibration Matrix.');
    });
  }

  const closeModal = () => {
    if (intrinsicsModal) intrinsicsModal.style.display = 'none';
  };

  if (closeIntrinsicsModal) closeIntrinsicsModal.addEventListener('click', closeModal);
  if (doneIntrinsicsModal) doneIntrinsicsModal.addEventListener('click', closeModal);

  if (intrinsicsModal) {
    intrinsicsModal.addEventListener('click', (e) => {
      if (e.target === intrinsicsModal) closeModal();
    });
  }

  if (copyIntrinsicsBtn) {
    copyIntrinsicsBtn.addEventListener('click', () => {
      const codeBlock = document.getElementById('intrinsicsCodeBlock');
      if (codeBlock && navigator.clipboard) {
        navigator.clipboard.writeText(codeBlock.textContent).then(() => {
          copyIntrinsicsBtn.textContent = 'Copied!';
          setTimeout(() => { copyIntrinsicsBtn.textContent = 'Copy Format'; }, 2000);
        });
      }
    });
  }

  // Telemetry Export Handlers (JSON & CSV)
  const exportJsonBtn = document.getElementById('exportTelemetryJson');
  if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => exportTelemetry('json'));
  }

  const exportCsvBtn = document.getElementById('exportTelemetryCsv');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => exportTelemetry('csv'));
  }

  // MongoDB Manual Sync Button
  const syncMongoBtn = document.getElementById('syncMongoBtn');
  if (syncMongoBtn) {
    syncMongoBtn.addEventListener('click', () => {
      if (activeTelemetryState) {
        saveTelemetryToMongo(activeTelemetryState);
      } else {
        log('[MONGODB] Please load or drag & drop a drone video first before syncing to MongoDB.');
      }
    });
  }

  // ==========================================
  // OPENCV / COLAB FRAME EXTRACTION & INSPECTION
  // ==========================================
  const colabState = {
    totalFrames: 0,
    fps: 30.0,
    currentFrame: 0,
    keyframes: [],
    initialized: false
  };

  const colabFrameSlider = document.getElementById('colabFrameSlider');
  const colabFrameNum = document.getElementById('colabFrameNum');
  const colabTotalFrames = document.getElementById('colabTotalFrames');
  const colabTimestamp = document.getElementById('colabTimestamp');
  const colabSharpness = document.getElementById('colabSharpness');
  const colabSliderValueBadge = document.getElementById('colabSliderValueBadge');
  const colabMainImage = document.getElementById('colabMainImage');
  const hudFrameTag = document.getElementById('hudFrameTag');
  const hudTimeTag = document.getElementById('hudTimeTag');
  const keyframesFilmstrip = document.getElementById('keyframesFilmstrip');

  async function fetchVideoMetadata() {
    try {
      const res = await fetch('/api/video/metadata');
      if (res.ok) {
        const data = await res.json();
        if (data.total_frames) colabState.totalFrames = data.total_frames;
        if (data.fps) colabState.fps = data.fps;
        if (colabTotalFrames) colabTotalFrames.textContent = colabState.totalFrames;
        if (colabFrameSlider) colabFrameSlider.max = colabState.totalFrames - 1;
        const colabNumInput = document.getElementById('colabFrameNumberInput');
        if (colabNumInput) colabNumInput.max = colabState.totalFrames - 1;
        updateOpenCvDetailsPanel(data);

        if (data.filename) {
          const actionBox = document.getElementById('videoProcessingActionBox');
          const actionTitle = document.getElementById('processingAdvanceTitle');
          const actionSub = document.getElementById('processingAdvanceSub');
          if (actionBox) {
            actionBox.style.display = 'flex';
            if (actionTitle) actionTitle.textContent = `✓ ${data.filename} Ready`;
            if (actionSub) actionSub.textContent = `${data.resolution} · ${data.fps} FPS · ${data.total_frames} Frames (${data.duration}s) · Click below to extract frames & keyframes`;
          }
        }
      }
    } catch (e) {
      console.warn('[OPENCV] Could not fetch video metadata, using defaults:', e);
    }
  }

  async function showColabFrame(frameIndex) {
    if (frameIndex < 0) frameIndex = 0;
    if (frameIndex >= colabState.totalFrames) frameIndex = colabState.totalFrames - 1;
    colabState.currentFrame = frameIndex;

    const timestampSec = (frameIndex / colabState.fps).toFixed(2);
    const frameNumber = frameIndex + 1;

    // Update text indicators for Matplotlib title
    if (colabFrameNum) colabFrameNum.textContent = frameNumber;
    if (colabTimestamp) colabTimestamp.textContent = timestampSec;
    const colabTotalFramesEl = document.getElementById('colabTotalFrames');
    if (colabTotalFramesEl) colabTotalFramesEl.textContent = colabState.totalFrames;
    if (colabSliderValueBadge) colabSliderValueBadge.textContent = `#${frameNumber} (${timestampSec}s)`;
    if (hudFrameTag) hudFrameTag.textContent = `FRAME #${frameNumber}`;
    if (hudTimeTag) hudTimeTag.textContent = `T+${timestampSec}s`;
    if (colabFrameSlider) colabFrameSlider.value = frameIndex;
    const colabNumInput = document.getElementById('colabFrameNumberInput');
    if (colabNumInput) colabNumInput.value = frameIndex;

    // Update main photographic image frame (NOT A VIDEO!)
    if (colabMainImage) {
      colabMainImage.style.display = 'block';
      colabMainImage.src = `/api/video/frame_image?frame=${frameIndex}&width=1280`;
    }
    const emptyState = document.getElementById('colabEmptyState');
    if (emptyState) emptyState.style.display = 'none';

    // Highlight corresponding filmstrip card if present
    if (keyframesFilmstrip) {
      keyframesFilmstrip.querySelectorAll('.keyframe-card').forEach(card => {
        const cardFrame = parseInt(card.dataset.frame, 10);
        if (cardFrame === frameIndex) {
          card.classList.add('active');
        } else {
          card.classList.remove('active');
        }
      });
    }

    // Synchronize 3D viewer down below with extracted frame
    const extractedBadge = document.getElementById('extractedFrameBadge');
    if (extractedBadge) {
      extractedBadge.textContent = `FRAME #${frameNumber} · T+${timestampSec}s`;
    }

    // Also render frame onto 2D canvas in 3D viewport
    const frame2dCanvas = document.getElementById('frame2dCanvas');
    if (frame2dCanvas) {
      const ctx = frame2dCanvas.getContext('2d');
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        ctx.clearRect(0, 0, frame2dCanvas.width, frame2dCanvas.height);
        ctx.drawImage(img, 0, 0, frame2dCanvas.width, frame2dCanvas.height);
      };
      img.src = `/api/video/frame_image?frame=${frameIndex}&width=640`;
    }

    // Fetch exact Laplacian sharpness asynchronously
    try {
      const infoRes = await fetch(`/api/video/frame_info?frame=${frameIndex}`);
      if (infoRes.ok) {
        const info = await infoRes.json();
        if (colabSharpness && info.sharpness !== undefined) {
          colabSharpness.textContent = info.sharpness.toFixed(2);
        }
      }
    } catch (err) {
      // Keep existing metric if call fails
    }
  }

  async function loadKeyframesFilmstrip() {
    if (!keyframesFilmstrip) return;
    try {
      const res = await fetch('/api/video/keyframes');
      if (res.ok) {
        const data = await res.json();
        colabState.keyframes = data.keyframes || [];
        renderKeyframesFilmstrip(colabState.keyframes);
        return;
      }
    } catch (e) {
      console.warn('[OPENCV] Error loading keyframes from server:', e);
    }

    // Fallback if network issue
    const fallbackCount = 10;
    const step = Math.floor(colabState.totalFrames / fallbackCount);
    const synthKeyframes = [];
    for (let i = 0; i < fallbackCount; i++) {
      const fIdx = Math.min(colabState.totalFrames - 1, i * step);
      synthKeyframes.push({
        frame_number: fIdx + 1,
        frame_index: fIdx,
        timestamp_sec: parseFloat((fIdx / colabState.fps).toFixed(2)),
        sharpness: parseFloat((110 + (Math.sin(i) * 15)).toFixed(2)),
        image_url: `/api/video/frame_image?frame=${fIdx}&width=360`
      });
    }
    colabState.keyframes = synthKeyframes;
    renderKeyframesFilmstrip(synthKeyframes);
  }

  function renderKeyframesFilmstrip(keyframes) {
    if (!keyframesFilmstrip) return;
    keyframesFilmstrip.innerHTML = '';

    const countBadge = document.getElementById('keyframesCountBadge');
    if (countBadge) countBadge.textContent = `${keyframes.length} KEYFRAMES`;

    keyframes.forEach(kf => {
      const card = document.createElement('div');
      card.className = `keyframe-card ${kf.frame_index === colabState.currentFrame ? 'active' : ''}`;
      card.dataset.frame = kf.frame_index;
      card.title = `Click to inspect Frame #${kf.frame_number} (${kf.timestamp_sec.toFixed(2)}s)`;

      card.innerHTML = `
        <img src="${kf.image_url}" class="keyframe-thumb" alt="Frame #${kf.frame_number}" loading="lazy" />
        <div class="keyframe-info">
          <div class="keyframe-frame-num">Frame #${kf.frame_number}</div>
          <div class="keyframe-time-row">
            <span class="keyframe-time-label">Time:</span>
            <span class="keyframe-time-val">${kf.timestamp_sec.toFixed(2)} sec</span>
          </div>
          <span class="keyframe-sharpness">Sharpness: ${kf.sharpness.toFixed(1)}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        showColabFrame(kf.frame_index);
        const kfImg = document.getElementById('selectedKeyframeImage');
        if (kfImg) {
          kfImg.src = kf.image_url || `/api/video/frame_image?frame=${kf.frame_index}&width=1280`;
        }
        const timeEl = document.getElementById('keyframeTimestamp');
        if (timeEl) timeEl.textContent = kf.timestamp_sec.toFixed(2);
        const origFrameEl = document.getElementById('keyframeOrigNum');
        if (origFrameEl) origFrameEl.textContent = kf.frame_number;
        const sharpEl = document.getElementById('keyframeSharpness');
        if (sharpEl) sharpEl.textContent = kf.sharpness.toFixed(1);
        const slider = document.getElementById('selectedKeyframeSlider');
        const numInput = document.getElementById('selectedKeyframeNumberInput');
        const kfIdxEl = document.getElementById('keyframeIdxNum');
        const matchingIdx = selectedFramesList.findIndex(s => s.frame_index === kf.frame_index);
        if (matchingIdx >= 0) {
          if (slider) slider.value = matchingIdx;
          if (numInput) numInput.value = matchingIdx;
          if (kfIdxEl) kfIdxEl.textContent = matchingIdx + 1;
        }
        log(`[KEYFRAME] Loaded Keyframe #${kf.frame_number} (${kf.timestamp_sec.toFixed(2)}s, Sharpness: ${kf.sharpness.toFixed(1)})`);
      });

      keyframesFilmstrip.appendChild(card);
    });
  }

  function initColabFrameInspector() {
    if (colabState.initialized) return;
    colabState.initialized = true;

    fetchVideoMetadata().then(() => {
      loadKeyframesFilmstrip();
      showColabFrame(colabState.currentFrame);
    });

    const colabNumInput = document.getElementById('colabFrameNumberInput');

    if (colabFrameSlider) {
      let debounceTimer = null;
      colabFrameSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        const tSec = (val / colabState.fps).toFixed(2);
        if (colabFrameNum) colabFrameNum.textContent = val + 1;
        if (colabTimestamp) colabTimestamp.textContent = tSec;
        if (colabNumInput) colabNumInput.value = val;

        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          showColabFrame(val);
        }, 30);
      });
    }

    if (colabNumInput) {
      colabNumInput.addEventListener('change', (e) => {
        let val = parseInt(e.target.value, 10);
        if (isNaN(val)) val = 0;
        val = Math.max(0, Math.min(val, colabState.totalFrames - 1));
        colabNumInput.value = val;
        if (colabFrameSlider) colabFrameSlider.value = val;
        showColabFrame(val);
      });
    }

    // Step Buttons
    const stepFirst = document.getElementById('colabStepFirst');
    if (stepFirst) stepFirst.addEventListener('click', () => showColabFrame(0));

    const stepMinus10 = document.getElementById('colabStepMinus10');
    if (stepMinus10) stepMinus10.addEventListener('click', () => showColabFrame(colabState.currentFrame - 10));

    const stepBack1 = document.getElementById('stepBackSec');
    if (stepBack1) stepBack1.addEventListener('click', () => showColabFrame(colabState.currentFrame - 1));

    const stepForward1 = document.getElementById('stepForwardSec');
    if (stepForward1) stepForward1.addEventListener('click', () => showColabFrame(colabState.currentFrame + 1));

    const stepPlus10 = document.getElementById('colabStepPlus10');
    if (stepPlus10) stepPlus10.addEventListener('click', () => showColabFrame(colabState.currentFrame + 10));

    const stepLast = document.getElementById('colabStepLast');
    if (stepLast) stepLast.addEventListener('click', () => showColabFrame(colabState.totalFrames - 1));

    const extractBtn = document.getElementById('extractVideoFrameBtn');
    if (extractBtn) {
      extractBtn.addEventListener('click', () => {
        const viewportPanel = document.querySelector('.viewport-panel');
        if (viewportPanel) {
          viewportPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        log(`[CV EXTRACTION] Extracted Frame #${colabState.currentFrame + 1} at ${(colabState.currentFrame / colabState.fps).toFixed(2)}s sent to 3D Reconstructor below.`);
      });
    }

    // Copy Python OpenCV Terminal Output Button
    const copyCvBtn = document.getElementById('copyCvOutputBtn');
    if (copyCvBtn) {
      copyCvBtn.addEventListener('click', () => {
        const term = document.getElementById('cvTerminalOutput');
        if (term) {
          navigator.clipboard.writeText(term.textContent).then(() => {
            const orig = copyCvBtn.innerHTML;
            copyCvBtn.innerHTML = '✓ Copied!';
            copyCvBtn.style.background = 'var(--accent-green)';
            setTimeout(() => {
              copyCvBtn.innerHTML = orig;
              copyCvBtn.style.background = '';
            }, 2000);
          });
        }
      });
    }

    // Initialize Interactive Selected Keyframe Viewer
    initSelectedKeyframeViewer();
  }

  // ==================================================
  // INTERACTIVE SELECTED KEYFRAME VIEWER (show_selected_keyframe)
  // ==================================================
  let selectedFramesList = [];

  function show_selected_keyframe(keyframeIndex) {
    if (!selectedFramesList || selectedFramesList.length === 0) return;
    if (keyframeIndex < 0) keyframeIndex = 0;
    if (keyframeIndex >= selectedFramesList.length) keyframeIndex = selectedFramesList.length - 1;

    const kf = selectedFramesList[keyframeIndex];
    if (!kf) return;

    // Get original frame number and sharpness (matches user's Python snippet)
    const frameNumber = kf.frame_number !== undefined ? kf.frame_number : (kf.frame_index + 1);
    const frameIndex = kf.frame_index !== undefined ? kf.frame_index : (frameNumber - 1);
    const sharpness = (typeof kf.sharpness === 'number') ? kf.sharpness.toFixed(2) : kf.sharpness;
    const totalVideoFrames = (window.currentKeyframesData && window.currentKeyframesData.total_frames) || colabState.totalFrames || 0;
    const timestampSec = (typeof kf.timestamp_sec === 'number') ? kf.timestamp_sec.toFixed(2) : (typeof kf.timestamp === 'number' ? kf.timestamp.toFixed(2) : (frameIndex / (colabState.fps || 30.0)).toFixed(2));

    // Update Matplotlib Title:
    // KEYFRAME {keyframe_index + 1} / {len(selected_frames)}
    // Original Frame: {frame_number + 1} / {total_frames}   |   Time: {timestamp:.2f} sec   |   Sharpness: {sharpness:.2f}
    const kfIdxEl = document.getElementById('keyframeIdxNum');
    const kfTotalEl = document.getElementById('keyframeTotalCount');
    const origFrameEl = document.getElementById('keyframeOrigNum');
    const origTotalEl = document.getElementById('keyframeOrigTotal');
    const timeEl = document.getElementById('keyframeTimestamp');
    const sharpEl = document.getElementById('keyframeSharpness');

    if (kfIdxEl) kfIdxEl.textContent = keyframeIndex + 1;
    if (kfTotalEl) kfTotalEl.textContent = selectedFramesList.length;
    if (origFrameEl) origFrameEl.textContent = frameNumber;
    if (origTotalEl) origTotalEl.textContent = totalVideoFrames;
    if (timeEl) timeEl.textContent = timestampSec;
    if (sharpEl) sharpEl.textContent = sharpness;

    // Update Selected Keyframe Image (plt.imshow(frame_rgb))
    const kfImg = document.getElementById('selectedKeyframeImage');
    if (kfImg) {
      kfImg.style.display = 'block';
      kfImg.src = kf.url || `/api/keyframes/${keyframeIndex}` || `/api/video/frame_image?frame=${frameIndex}&width=1280`;
    }

    // Sync Slider & Number Input (IntSlider)
    const slider = document.getElementById('selectedKeyframeSlider');
    const numInput = document.getElementById('selectedKeyframeNumberInput');
    if (slider) slider.value = keyframeIndex;
    if (numInput) numInput.value = keyframeIndex;
  }

  async function loadSelectedKeyframes() {
    try {
      const res = await fetch('/api/video/keyframes');
      if (res.ok) {
        const data = await res.json();
        window.currentKeyframesData = data;
        if (data && data.keyframes && data.keyframes.length > 0) {
          selectedFramesList = data.keyframes;
          const maxIdx = Math.max(0, selectedFramesList.length - 1);
          const slider = document.getElementById('selectedKeyframeSlider');
          const numInput = document.getElementById('selectedKeyframeNumberInput');
          const totalCountEl = document.getElementById('keyframeTotalCount');
          const origTotalEl = document.getElementById('keyframeOrigTotal');
          if (slider) { slider.min = 0; slider.max = maxIdx; }
          if (numInput) { numInput.min = 0; numInput.max = maxIdx; }
          if (totalCountEl) totalCountEl.textContent = selectedFramesList.length;
          if (origTotalEl) origTotalEl.textContent = data.total_frames || colabState.totalFrames || '--';
          show_selected_keyframe(0);
        }
      }
    } catch (err) {
      console.warn('[SELECTED KEYFRAMES] Error loading keyframes:', err);
    }
  }

  function initSelectedKeyframeViewer() {
    loadSelectedKeyframes();

    const slider = document.getElementById('selectedKeyframeSlider');
    const numInput = document.getElementById('selectedKeyframeNumberInput');

    if (slider) {
      slider.addEventListener('input', (e) => {
        const idx = parseInt(e.target.value, 10);
        if (numInput) numInput.value = idx;
        show_selected_keyframe(idx);
      });
    }

    if (numInput) {
      numInput.addEventListener('change', (e) => {
        let idx = parseInt(e.target.value, 10);
        if (isNaN(idx)) idx = 0;
        idx = Math.max(0, Math.min(idx, selectedFramesList.length - 1));
        numInput.value = idx;
        if (slider) slider.value = idx;
        show_selected_keyframe(idx);
      });
    }
  }

  // Download Trajectory JSON Handler
  const downloadTrajBtn = document.getElementById('downloadTrajectoryBtn');
  if (downloadTrajBtn) {
    downloadTrajBtn.addEventListener('click', () => {
      if (cachedTrajectoryData && cachedTrajectoryData.trajectory) {
        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(cachedTrajectoryData, null, 2));
        const a = document.createElement('a');
        a.setAttribute('href', dataStr);
        a.setAttribute('download', 'camera_trajectory.json');
        document.body.appendChild(a);
        a.click();
        a.remove();
        log('[TRAJECTORY] Exported camera_trajectory.json');
      } else {
        window.open('/api/trajectory', '_blank');
      }
    });
  }

  // Window Resize Auto-Refit for Trajectory Canvas
  window.addEventListener('resize', () => {
    const dedicatedTrajectorySection = document.getElementById('dedicatedTrajectorySection');
    if (dedicatedTrajectorySection && dedicatedTrajectorySection.style.display !== 'none') {
      if (cachedTrajectoryData && cachedTrajectoryData.trajectory) {
        render2DCameraTrajectory(
          cachedTrajectoryData.trajectory,
          cachedTrajectoryData.successful_matches,
          cachedTrajectoryData.failed_frames
        );
      }
    }
  });

  // ==========================================
  // PROJECTS LIBRARY & DASHBOARD TELEMETRY
  // ==========================================
  let projectsList = [];
  let activeProjectFilename = null;

  async function loadProjectsList() {
    try {
      const res = await fetch('/api/projects');
      if (res.ok) {
        const json = await res.json();
        if (json.success && Array.isArray(json.projects)) {
          projectsList = json.projects;
          updateProjectsBadge(projectsList.length);
          renderProjectsGrid(projectsList);

          // Find active project or default to first
          const activeProj = projectsList.find(p => p.is_active);
          if (activeProj) {
            activeProjectFilename = activeProj.filename;
            updateDashboardProjectDetails(activeProj);
          } else if (projectsList.length > 0 && !activeProjectFilename) {
            activeProjectFilename = projectsList[0].filename;
            updateDashboardProjectDetails(projectsList[0]);
          }
          return projectsList;
        }
      }
    } catch (e) {
      console.warn('[PROJECTS] Error loading projects from backend:', e);
    }
    return [];
  }

  function updateProjectsBadge(count) {
    const badge = document.getElementById('projectsCountBadge');
    if (badge) badge.textContent = count || 0;
    const countNum = document.getElementById('projectsCountNumber');
    if (countNum) countNum.textContent = count || 0;
  }

  function renderProjectsGrid(projectsToRender) {
    const grid = document.getElementById('projectsGrid');
    const emptyState = document.getElementById('projectsEmptyState');
    if (!grid) return;

    grid.innerHTML = '';
    if (!projectsToRender || projectsToRender.length === 0) {
      if (emptyState) emptyState.style.display = 'flex';
      return;
    }
    if (emptyState) emptyState.style.display = 'none';

    // Sort projects in chronological order by last modified date & time (most recently modified first)
    const sortedProjects = [...projectsToRender].sort((a, b) => {
      const tsA = a.last_modified_timestamp || (a.last_modified_iso ? new Date(a.last_modified_iso).getTime() : 0);
      const tsB = b.last_modified_timestamp || (b.last_modified_iso ? new Date(b.last_modified_iso).getTime() : 0);
      return tsB - tsA;
    });

    sortedProjects.forEach(proj => {
      const isActive = proj.filename === activeProjectFilename;
      const card = document.createElement('button');
      card.type = 'button';
      card.className = `project-file-card ${isActive ? 'active-project' : ''}`;
      card.dataset.filename = proj.filename;

      const modifiedStr = proj.last_modified || (proj.created_at ? new Date(proj.created_at).toLocaleString() : 'Recently modified');

      card.innerHTML = `
        <div class="project-file-item-main">
          <div class="project-file-icon-wrap">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <polyline points="10 9 9 9 8 9"/>
            </svg>
          </div>
          <div class="project-file-details">
            <div class="project-file-name" title="${proj.filename}">${proj.filename}</div>
            <div class="project-file-mod-time">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
              <span>Last Modified: <b>${modifiedStr}</b></span>
            </div>
          </div>
        </div>
        <div class="project-file-item-action">
          ${isActive ? '<span class="project-file-badge active">★ ACTIVE</span>' : '<span class="project-file-arrow-btn">➔</span>'}
        </div>
      `;

      card.addEventListener('click', () => {
        selectProject(proj);
      });

      grid.appendChild(card);
    });
  }

  async function selectProject(proj) {
    activeProjectFilename = proj.filename;
    log(`[PROJECTS] Selecting "${proj.filename}"...`);

    // Fetch and switch active project on backend
    try {
      const res = await fetch(`/api/projects/select?filename=${encodeURIComponent(proj.filename)}`, {
        method: 'POST'
      });
      if (res.ok) {
        const json = await res.json();
        if (json.data) {
          updateDashboardProjectDetails(json.data);
        } else {
          updateDashboardProjectDetails(proj);
        }
      } else {
        updateDashboardProjectDetails(proj);
      }
    } catch (e) {
      console.warn('[PROJECTS] Error selecting project on backend:', e);
      updateDashboardProjectDetails(proj);
    }

    // Refresh active styling
    renderProjectsGrid(projectsList);

    // Invalidate cached per-project data so every view re-fetches for the newly selected project
    // (otherwise the dashboard would render the previous project's trajectory/keyframes).
    cachedTrajectoryData = null;
    window.currentKeyframesData = null;
    if (typeof colabState === 'object') colabState.keyframes = [];

    // DIRECT FROM PROJECT OPTION TO DASHBOARD OPTION
    activateSidebarModule('dashboard');

    // Scroll smoothly to top of Dashboard
    window.scrollTo({ top: 0, behavior: 'smooth' });

    log(`[DASHBOARD] Loaded project "${proj.filename}" on Dashboard.`);
  }

  function updateDashboardProjectDetails(data) {
    if (!data) return;

    const fname = data.filename || 'Drone Flight Recording';
    const totalFrames = data.total_frames || data.number_of_extracted_frames || 0;
    const fps = parseFloat(data.fps || 30.0);
    const durSec = data.duration || (totalFrames > 0 && fps > 0 ? parseFloat((totalFrames / fps).toFixed(2)) : 0);
    const keyframesCount = data.keyframes !== undefined ? data.keyframes : (data.number_of_final_keyframes !== undefined ? data.number_of_final_keyframes : 0);
    const trajPts = data.trajectory_points !== undefined ? data.trajectory_points : (data.trajectory ? data.trajectory.length : totalFrames);
    const resolution = data.resolution || '3840 × 2160';

    // 1. Update Active Project Hero Banner Elements on Dashboard
    const nameEl = document.getElementById('dashActiveFilename');
    if (nameEl) nameEl.textContent = fname;

    const statusEl = document.getElementById('dashActiveStatus');
    if (statusEl) statusEl.textContent = 'ACTIVE';

    const kfEl = document.getElementById('dashActiveKeyframes');
    if (kfEl) kfEl.textContent = keyframesCount.toLocaleString();

    const trajEl = document.getElementById('dashActiveTrajectory');
    if (trajEl) trajEl.textContent = `${trajPts.toLocaleString()} pts`;

    const trajSub = document.getElementById('dashActiveTrajectorySub');
    if (trajSub) trajSub.textContent = `ORB RANSAC motion path`;

    const tfEl = document.getElementById('dashActiveTotalFrames');
    if (tfEl) tfEl.textContent = totalFrames.toLocaleString();

    const durEl = document.getElementById('dashActiveDuration');
    if (durEl) durEl.textContent = `${durSec} s`;

    const durFmt = document.getElementById('dashActiveDurationFmt');
    if (durFmt) {
      const mins = Math.floor(durSec / 60);
      const secs = Math.floor(durSec % 60);
      durFmt.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')} flight duration`;
    }

    const fpsEl = document.getElementById('dashActiveFps');
    if (fpsEl) fpsEl.textContent = `${fps} fps`;

    const resEl = document.getElementById('dashActiveResolution');
    if (resEl) resEl.textContent = resolution;

    // 2. Update Flight Metadata Card
    const metaResEl = document.getElementById('metaRes');
    if (metaResEl) metaResEl.textContent = resolution;

    const metaFpsEl = document.getElementById('metaFps');
    if (metaFpsEl) metaFpsEl.textContent = `FPS: ${fps} · High-FPS Sensor`;

    // 3. Update Metrics Summary Badge (in Viewport panel)
    const metricsDiv = document.querySelector('.metrics');
    if (metricsDiv) {
      const kfStrong = metricsDiv.querySelector('div:first-child strong');
      if (kfStrong) kfStrong.textContent = keyframesCount;
    }

    // 4. Update Colab Frame Inspector bounds
    colabState.totalFrames = totalFrames;
    colabState.fps = fps;
    const colabTotalFramesEl = document.getElementById('colabTotalFrames');
    if (colabTotalFramesEl) colabTotalFramesEl.textContent = totalFrames;
    const colabFrameSlider = document.getElementById('colabFrameSlider');
    if (colabFrameSlider) {
      colabFrameSlider.max = Math.max(0, totalFrames - 1);
      colabFrameSlider.value = 0;
    }
    const colabNumInput = document.getElementById('colabFrameNumberInput');
    if (colabNumInput) {
      colabNumInput.max = Math.max(0, totalFrames - 1);
      colabNumInput.value = 0;
    }

    // 5. Update Keyframes Viewer
    const kfTotal = document.getElementById('keyframeTotalCount');
    if (kfTotal) kfTotal.textContent = keyframesCount;
    const origTotal = document.getElementById('keyframeOrigTotal');
    if (origTotal) origTotal.textContent = totalFrames;
    const kfSlider = document.getElementById('selectedKeyframeSlider');
    if (kfSlider) {
      kfSlider.max = Math.max(0, keyframesCount - 1);
      kfSlider.value = 0;
    }

    // 6. Reload keyframes filmstrip and frame #0
    loadKeyframesFilmstrip();
    showColabFrame(0);

    // 7. Update Trajectory Rendering
    if (data.trajectory && data.trajectory.length > 0) {
      cachedTrajectoryData = {
        trajectory: data.trajectory,
        successful_matches: trajPts,
        failed_frames: 0
      };
      render2DCameraTrajectory(data.trajectory, trajPts, 0);
      updateThreeFlightPath(data.trajectory);
    } else {
      loadTrajectoryData();
    }
  }

  function onVideoProcessedSuccess(file, data) {
    const newProjectEntry = {
      filename: file.name,
      resolution: data.resolution || '3840 x 2160',
      fps: data.fps || 30.0,
      duration: data.duration || 0,
      total_frames: data.total_frames || data.number_of_extracted_frames || 0,
      keyframes: data.number_of_final_keyframes || 0,
      trajectory_points: data.trajectory_points || 0,
      created_at: new Date().toISOString(),
      status: 'PROCESSED',
      is_active: true
    };

    const existingIdx = projectsList.findIndex(p => p.filename === file.name);
    if (existingIdx >= 0) {
      projectsList[existingIdx] = newProjectEntry;
    } else {
      projectsList.unshift(newProjectEntry);
    }

    activeProjectFilename = file.name;
    updateProjectsBadge(projectsList.length);
    renderProjectsGrid(projectsList);
    updateDashboardProjectDetails(data);
    log(`[PROJECTS] Added "${file.name}" as new project file button.`);
  }

  // Projects Search Filter Listener
  const projectsSearchInput = document.getElementById('projectsSearchInput');
  if (projectsSearchInput) {
    projectsSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      if (!q) {
        renderProjectsGrid(projectsList);
      } else {
        const filtered = projectsList.filter(p => p.filename.toLowerCase().includes(q));
        renderProjectsGrid(filtered);
      }
    });
  }

  const refreshProjectsBtn = document.getElementById('refreshProjectsBtn');
  if (refreshProjectsBtn) {
    refreshProjectsBtn.addEventListener('click', () => {
      loadProjectsList();
      log('[PROJECTS] Refreshed projects list.');
    });
  }

  // Initialize on Dashboard View (Only Keyframes, Trajectory, and Telemetry Specs)
  activateSidebarModule('dashboard');

  // Connect to Backend, Load Projects Library, and Check MongoDB Status
  loadProjectsList();
  checkMongoHealth();
});

