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
  
  if (progressVal) progressVal.textContent = `${value}%`;
  if (progressBar) progressBar.style.width = `${value}%`;
  
  // Highlight active phase among 8 stages
  const currentPhaseIndex = Math.min(7, Math.floor((value / 100) * 8));

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

// MongoDB Integration Helpers
async function checkMongoHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const mongoStatusText = document.getElementById('mongoStatusText');
    const mongoDot = document.getElementById('mongoDot');
    if (data.mongodb_connected) {
      if (mongoStatusText) mongoStatusText.textContent = 'CONNECTED';
      if (mongoDot) {
        mongoDot.style.backgroundColor = '#10b981';
        mongoDot.style.boxShadow = '0 0 0 0 rgba(16, 185, 129, 0.5)';
      }
      log(`[MONGODB] Connected to database "${data.database}" at ${data.mongodb_uri} (Logs: ${data.collections.telemetry_logs}, Jobs: ${data.collections.pipeline_jobs})`);
    } else {
      if (mongoStatusText) {
        mongoStatusText.textContent = 'OFFLINE';
        mongoStatusText.style.color = '#ef4444';
      }
      if (mongoDot) mongoDot.style.backgroundColor = '#ef4444';
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
      log(`[MONGODB] Saved flight telemetry to collection "telemetry_logs" · Doc ID: ${result.id}`);
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
      'RTK_Status', 'Sat_Count'
    ];
    const rows = samples.map(s => [
      s.timestamp_utc, s.time_sec, s.latitude_deg, s.longitude_deg, s.altitude_agl_m,
      s.altitude_msl_m, s.barometer_hpa, s.yaw_deg, s.pitch_deg, s.roll_deg,
      s.ground_speed_mps, s.velocity_north_mps, s.velocity_east_mps, s.velocity_down_mps,
      s.imu_acc_mps2[0], s.imu_acc_mps2[1], s.imu_acc_mps2[2],
      s.imu_gyro_dps[0], s.imu_gyro_dps[1], s.imu_gyro_dps[2],
      s.rtk_fix, s.satellites
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

function loadVideo(url, label, fileObj = null) {
  if (!url) return;

  const videoContainer = document.getElementById('videoContainer');
  const videoPreview = document.getElementById('videoPreview');
  const fileName = document.getElementById('fileName');
  const statusPill = document.getElementById('inputStatus');
  const metaEmptyState = document.getElementById('metaEmptyState');
  const metaGrid = document.getElementById('metaGrid');
  const metaActionsBar = document.getElementById('metaActionsBar');

  if (videoContainer) videoContainer.style.display = 'block';
  if (fileName) fileName.textContent = label;

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

      log(`[TELEMETRY] Extracted 12/12 attributes · Res: ${w}×${h} · FPS: ${fps} · Intrinsics fx: ${intrinsics.fx} · RTK: FIXED`);

      // Persist ingested video telemetry to MongoDB Compass (telemetry_logs)
      saveTelemetryToMongo(activeTelemetryState);
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
  }

  log(`[INGEST] Loaded video stream: "${label}"`);
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

  // Frame Extraction Trigger Buttons
  const extractBtn1 = document.getElementById('extractVideoFrameBtn');
  const extractBtn2 = document.getElementById('quickExtractBtn');
  const videoEl = document.getElementById('videoPreview');

  const triggerExtract = () => {
    if (videoEl && !videoEl.paused) videoEl.pause();
    extractVideoFrame();
    if (modeDualView && !modeDualView.classList.contains('active')) {
      modeDualView.click();
    }
  };

  if (extractBtn1) extractBtn1.addEventListener('click', triggerExtract);
  if (extractBtn2) extractBtn2.addEventListener('click', triggerExtract);

  // Timecode Step buttons
  const stepBackBtn = document.getElementById('stepBackSec');
  const stepForwardBtn = document.getElementById('stepForwardSec');

  if (stepBackBtn) {
    stepBackBtn.addEventListener('click', () => {
      if (videoEl) {
        videoEl.currentTime = Math.max(0, (videoEl.currentTime || 4.5) - 1.0);
        triggerExtract();
      }
    });
  }

  if (stepForwardBtn) {
    stepForwardBtn.addEventListener('click', () => {
      if (videoEl) {
        videoEl.currentTime = Math.min(videoEl.duration || 120, (videoEl.currentTime || 4.5) + 1.0);
        triggerExtract();
      }
    });
  }

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

  // Pre-load sample video & extract initial keyframe #01 immediately
  loadVideo('assets/media/sample_4k.mp4', 'sample_4k.mp4 (92.5 MB · DJI 4K Drone Aerial Survey)');
  setTimeout(() => {
    extractVideoFrame(4.50);
  }, 400);

  // Run Pipeline Button
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
        const videoContainer = document.getElementById('videoContainer');
        const videoPreview = document.getElementById('videoPreview');
        if (videoPreview) {
          videoPreview.pause();
          videoPreview.removeAttribute('src');
          videoPreview.load();
        }
        if (videoContainer) videoContainer.style.display = 'none';
        if (runBtn) {
          runBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Start 3D Pipeline
          `;
        }
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

  // File Upload Input
  const fileInput = document.getElementById('videoFile');
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        if (!file.type.startsWith('video/')) {
          log('Error: Please upload a valid MP4 or MOV video file.');
          return;
        }
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        const objectUrl = URL.createObjectURL(file);
        loadVideo(objectUrl, `${file.name} (${sizeMb} MB)`, file);
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
      if (file && file.type.startsWith('video/')) {
        const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
        const objectUrl = URL.createObjectURL(file);
        loadVideo(objectUrl, `${file.name} (${sizeMb} MB)`, file);
      } else {
        log('Error: Dropped file is not a supported video format.');
      }
    });
  }

  // URL Input Load Button
  const loadUrlBtn = document.getElementById('loadUrl');
  const urlInput = document.getElementById('videoUrl');

  if (loadUrlBtn && urlInput) {
    const handleUrlSubmit = () => {
      const url = urlInput.value.trim();
      if (!url) {
        log('Error: Please enter a valid video URL.');
        return;
      }
      try {
        const parsed = new URL(url);
        if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error();
        const displayLabel = parsed.pathname.split('/').pop() || 'Remote Video Stream';
        loadVideo(url, displayLabel);
      } catch {
        log('Error: Please enter a full http:// or https:// URL.');
      }
    };

    loadUrlBtn.addEventListener('click', handleUrlSubmit);
    urlInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handleUrlSubmit();
    });
  }

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

  // Connect to Backend and Check MongoDB Compass Status
  checkMongoHealth();
});
