// Aero3D · Stage 04 — Camera Pose (COLMAP) & Depth (Depth Anything V2) page.
//
// Round trip: export package -> run notebook in Colab -> import results. This file drives the page
// and a small Three.js viewer of the recovered camera poses + sparse point cloud.
// It shows itself when app.js sets data-active-view="depth-estimation" on #dashboardMainArea.
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const section = $('stage04Section');
  const area = $('dashboardMainArea');
  if (!section || !area) return;

  const VIEW_HEIGHT = 460;
  const state = { visible: false, importedAt: '', scene: null, viewer: null, cameraIdx: 0 };

  // ---------------------------------------------------------------- helpers ------------------
  async function api(path) {
    const res = await fetch(path);
    let body = null;
    try { body = await res.json(); } catch (e) { /* non-JSON error page */ }
    if (!res.ok) throw new Error((body && (body.detail || body.error)) || `Request failed (${res.status})`);
    return body;
  }

  function message(text, kind) {
    const el = $('s04Message');
    if (!text) { el.style.display = 'none'; return; }
    el.textContent = text;
    el.className = 's04-message' + (kind ? ` ${kind}` : '');
    el.style.display = 'block';
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function metricCard(label, badge, value, desc, badgeColor) {
    const card = el('div', 'dash-metric-card');
    const top = el('div', 'metric-top-row');
    top.append(el('span', 'metric-label', label), el('span', `metric-badge-pill ${badgeColor || 'cyan'}`, badge));
    card.append(top, el('strong', 'metric-value', value), el('small', 'metric-desc', desc));
    return card;
  }

  // ---------------------------------------------------------------- page state ---------------
  async function refresh() {
    message('');
    let status;
    try {
      status = await api('/api/stage04/status');
    } catch (err) {
      $('s04Project').textContent = 'No project';
      $('s04ExportInfo').textContent = 'No processed video yet.';
      $('s04ExportBtn').disabled = true;
      $('s04ImportBtn').disabled = true;
      $('s04Results').style.display = 'none';
      message(`${err.message} Upload a video and let Stage 02 finish first.`, 'error');
      return;
    }
    $('s04ImportBtn').disabled = false;
    $('s04Project').textContent = status.project;

    try {
      const info = await api('/api/stage04/export_info');
      const gaps = info.keyframe_gaps || {};
      let text = `${info.keyframes_on_disk} keyframes ready`;
      if (info.gap_fill_frames) text += ` + ${info.gap_fill_frames} gap-fill frames`;
      if (gaps.max_gap_frames !== undefined) text += ` · largest keyframe gap ${gaps.max_gap_frames} frames`;
      if (!info.ready) text = `Needs at least ${info.min_keyframes} keyframes (found ${info.keyframes_on_disk}).`;
      $('s04ExportInfo').textContent = text;
      $('s04ExportBtn').disabled = !info.ready;
    } catch (err) {
      $('s04ExportInfo').textContent = err.message;
      $('s04ExportBtn').disabled = true;
    }

    ['s04Step1', 's04Step2', 's04Step3'].forEach((id) => $(id).classList.toggle('done', !!status.results_imported));
    if (status.results_imported && status.summary) {
      renderSummary(status.summary);
      await loadScene();
    } else {
      $('s04Results').style.display = 'none';
      message('No results imported yet. Export the package, run the Colab notebook, then import stage04_results.zip.');
    }
  }

  function renderSummary(s) {
    state.importedAt = s.imported_at || '';
    $('s04Results').style.display = 'block';
    const verdict = $('s04Verdict');
    verdict.textContent = s.verdict;
    verdict.className = `s04-verdict ${s.verdict}`;
    const models = (s.colmap && s.colmap.num_models) || 1;
    $('s04VerdictNote').textContent = models > 1
      ? `${s.images.registered_any_model}/${s.images.total} images registered across ${models} reconstructions; showing the largest ` +
        `(${s.images.keyframes_registered}/${s.images.keyframes_total} keyframes) · ${s.reprojection_error_px} px mean reprojection error`
      : `${s.images.keyframes_registered}/${s.images.keyframes_total} keyframes registered` +
        (s.images.gap_fill_frames ? ` (+ ${s.images.gap_fill_registered}/${s.images.gap_fill_frames} gap-fill frames)` : '') +
        ` · ${s.reprojection_error_px} px mean reprojection error`;

    const warnings = $('s04Warnings');
    warnings.replaceChildren(...(s.warnings || []).map((w) => el('li', '', w)));

    const metrics = $('s04Metrics');
    const d = s.depth || {};
    const depthValue = d.available && d.holdout_rel_err_median !== null && d.holdout_rel_err_median !== undefined
      ? `${(d.holdout_rel_err_median * 100).toFixed(1)}% err` : (d.available ? 'n/a' : 'not included');
    const depthDesc = d.available
      ? `held-out depth error · ${String(d.model || '').split('/').pop()} · ${d.images_with_depth} maps`
      : 'run the depth cell in the notebook';
    metrics.replaceChildren(
      metricCard('REGISTERED', 'POSES', `${s.images.keyframes_registered} / ${s.images.keyframes_total}`,
        `${(s.images.keyframe_fraction * 100).toFixed(1)}% of keyframes have a 6-DoF pose` +
        (s.images.gap_fill_frames ? ` · +${s.images.gap_fill_frames} gap-fill frames` : ''), 'cyan'),
      metricCard('SPARSE POINTS', '3D', s.points.count.toLocaleString(),
        `mean track length ${s.points.mean_track_length}`, 'orange'),
      metricCard('REPROJECTION', 'ACCURACY', `${s.reprojection_error_px} px`, 'good is below 0.8 px', 'yellow'),
      metricCard('CAMERA', 'INTRINSICS', `f = ${s.camera.focal_px} px`,
        `${s.camera.model} · horizontal FOV ${s.camera.hfov_deg}°`, 'blue'),
      metricCard('DEPTH', 'DEPTH ANYTHING V2', depthValue, depthDesc, 'cyan'),
      metricCard('TRAJECTORY', 'PATH', `${s.trajectory.length_units} units`, 'COLMAP units, not metres', 'orange')
    );
    $('s04Units').textContent = s.units || '';
  }

  // ---------------------------------------------------------------- import / export -----------
  function exportPackage() {
    const a = document.createElement('a');
    a.href = '/api/stage04/export';
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
    message('Building the package... your download starts in a moment. Upload it in the Colab notebook.', 'ok');
  }

  function importResults(file) {
    if (!file) return;
    const btn = $('s04ImportBtn');
    btn.disabled = true;
    const form = new FormData();
    form.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/stage04/import');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) message(`Uploading ${file.name}... ${Math.round((e.loaded / e.total) * 100)}%`);
    };
    xhr.upload.onload = () => message('Validating and parsing the reconstruction...');
    xhr.onload = async () => {
      btn.disabled = false;
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
      if (xhr.status === 200 && body && body.summary) {
        message('Results imported.', 'ok');
        ['s04Step1', 's04Step2', 's04Step3'].forEach((id) => $(id).classList.add('done'));
        renderSummary(body.summary);
        await loadScene();
      } else {
        message((body && (body.detail || body.error)) || `Import failed (${xhr.status}).`, 'error');
      }
    };
    xhr.onerror = () => { btn.disabled = false; message('Upload failed: is the backend running?', 'error'); };
    xhr.send(form);
  }

  // ---------------------------------------------------------------- 3D viewer ----------------
  function ensureViewer() {
    if (state.viewer) return state.viewer;
    const canvas = $('s04Canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    camera.position.set(0, 0, 13);
    const orbit = new THREE.Group();      // user rotation
    const world = new THREE.Group();      // scene content, re-oriented so "up" is +Y
    orbit.add(world);
    scene.add(orbit);
    orbit.rotation.set(0.45, 0.6, 0);

    const v = { renderer, scene, camera, orbit, world, points: null, cams: null, path: null, highlight: null,
                dragging: false, last: [0, 0], raf: 0 };
    canvas.addEventListener('pointerdown', (e) => { v.dragging = true; v.last = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointerup', (e) => { v.dragging = false; canvas.releasePointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', (e) => {
      if (!v.dragging) return;
      orbit.rotation.y += (e.clientX - v.last[0]) * 0.006;
      orbit.rotation.x = Math.max(-1.5, Math.min(1.5, orbit.rotation.x + (e.clientY - v.last[1]) * 0.006));
      v.last = [e.clientX, e.clientY];
    });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      camera.position.z = Math.max(2, Math.min(40, camera.position.z * (1 + e.deltaY * 0.001)));
    }, { passive: false });
    state.viewer = v;
    resizeViewer();
    return v;
  }

  function resizeViewer() {
    const v = state.viewer;
    if (!v) return;
    const canvas = $('s04Canvas');
    const w = canvas.clientWidth || 800;
    v.renderer.setSize(w, VIEW_HEIGHT, false);
    v.camera.aspect = w / VIEW_HEIGHT;
    v.camera.updateProjectionMatrix();
  }

  function clearObject(obj) {
    if (!obj) return;
    obj.parent && obj.parent.remove(obj);
    obj.geometry && obj.geometry.dispose();
    obj.material && obj.material.dispose();
  }

  function frustumSegments(cam, apex, len, out) {
    const half = Math.tan((cam.hfov_deg * Math.PI) / 360) * len;
    const halfH = half / (cam.aspect || 1.78);
    const f = cam.forward, r = cam.right, d = cam.down;
    const corner = (sx, sy) => [0, 1, 2].map((i) => apex[i] + f[i] * len + r[i] * half * sx + d[i] * halfH * sy);
    const c = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
    for (let i = 0; i < 4; i++) {
      out.push(...apex, ...c[i]);              // apex -> corner
      out.push(...c[i], ...c[(i + 1) % 4]);    // image-plane rectangle
    }
  }

  function buildScene(scene) {
    if (!window.THREE) { message('Three.js failed to load, so the 3D view is unavailable.', 'error'); return; }
    const v = ensureViewer();
    [v.points, v.cams, v.path, v.highlight].forEach(clearObject);

    const P = scene.points, n = scene.point_count;
    const centers = scene.cameras.map((c) => c.center);
    const ref = n ? [] : centers;
    for (let i = 0; i < n; i += 1) ref.push([P[3 * i], P[3 * i + 1], P[3 * i + 2]]);
    // centre on the median and fit the 95th-percentile radius into a sphere of radius ~5
    const med = [0, 1, 2].map((k) => { const a = ref.map((p) => p[k]).sort((x, y) => x - y); return a[a.length >> 1]; });
    const dist = ref.map((p) => Math.hypot(p[0] - med[0], p[1] - med[1], p[2] - med[2])).sort((a, b) => a - b);
    const radius = dist[Math.floor(dist.length * 0.95)] || 1;
    const k = 5 / radius;
    const T = (p) => [(p[0] - med[0]) * k, (p[1] - med[1]) * k, (p[2] - med[2]) * k];

    const up = new THREE.Vector3(...scene.up).normalize();
    v.world.quaternion.setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
    v.world.userData = { T, k };

    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const p = T([P[3 * i], P[3 * i + 1], P[3 * i + 2]]);
      pos.set(p, 3 * i);
      col[3 * i] = scene.colors[3 * i] / 255; col[3 * i + 1] = scene.colors[3 * i + 1] / 255; col[3 * i + 2] = scene.colors[3 * i + 2] / 255;
    }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    pg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    v.points = new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.045, vertexColors: true, sizeAttenuation: true }));

    const seg = [], segCol = [], pathPts = [];
    const nCams = scene.cameras.length;
    scene.cameras.forEach((c, i) => {
      const apex = T(c.center);
      pathPts.push(new THREE.Vector3(...apex));
      const before = seg.length;
      frustumSegments(c, apex, 0.32, seg);
      const t = nCams > 1 ? i / (nCams - 1) : 0;
      const rgb = [0.2 + 0.8 * t, 0.6 - 0.2 * t, 1.0 - 0.9 * t];   // blue -> orange along the flight
      for (let j = before; j < seg.length; j += 3) segCol.push(...rgb);
    });
    const cg = new THREE.BufferGeometry();
    cg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(seg), 3));
    cg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(segCol), 3));
    v.cams = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ vertexColors: true }));
    v.path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pathPts),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }));
    [v.points, v.cams, v.path].forEach((o) => v.world.add(o));
    applyToggles();
    setHighlight(state.cameraIdx);

    $('s04SceneInfo').textContent =
      `${nCams} cameras · ${n.toLocaleString()} of ${scene.total_points.toLocaleString()} points shown`;
  }

  function applyToggles() {
    const v = state.viewer;
    if (!v) return;
    if (v.points) v.points.visible = $('s04ShowPoints').checked;
    if (v.cams) v.cams.visible = $('s04ShowCameras').checked;
    if (v.path) v.path.visible = $('s04ShowPath').checked;
  }

  function setHighlight(idx) {
    const v = state.viewer, scene = state.scene;
    if (!v || !scene || !scene.cameras[idx]) return;
    clearObject(v.highlight);
    const cam = scene.cameras[idx];
    const seg = [];
    frustumSegments(cam, v.world.userData.T(cam.center), 0.6, seg);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(seg), 3));
    v.highlight = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xffe600 }));
    v.world.add(v.highlight);
  }

  function loop() {
    const v = state.viewer;
    if (!state.visible || !v) { if (v) v.raf = 0; return; }
    v.renderer.render(v.scene, v.camera);
    v.raf = requestAnimationFrame(loop);
  }

  function startLoop() {
    const v = state.viewer;
    if (v && !v.raf) v.raf = requestAnimationFrame(loop);
  }

  // ---------------------------------------------------------------- keyframe + depth panel ---
  async function showDepth(idx) {
    const scene = state.scene;
    if (!scene || !scene.cameras.length) return;
    idx = Math.max(0, Math.min(scene.cameras.length - 1, idx));
    state.cameraIdx = idx;
    const cam = scene.cameras[idx];
    const frame = cam.frame_index;
    const stamp = encodeURIComponent(state.importedAt);
    $('s04DepthSlider').value = idx;
    $('s04DepthTitle').textContent = (cam.is_fill ? `GAP-FILL FRAME ${frame + 1}` : `KEYFRAME ${cam.keyframe_index + 1} · FRAME ${frame + 1}`) +
      ` · camera ${idx + 1} / ${scene.cameras.length}`;
    $('s04Rgb').src = cam.is_fill ? `/api/frames/${frame}?v=${stamp}` : `/api/keyframes/${cam.keyframe_index}?v=${stamp}`;
    const depthImg = $('s04DepthImg');
    depthImg.style.visibility = 'visible';
    depthImg.onerror = () => { depthImg.style.visibility = 'hidden'; $('s04DepthCaption').textContent = 'No depth map for this keyframe'; };
    depthImg.src = `/api/stage04/depth/${frame}?v=${stamp}`;
    $('s04DepthCaption').textContent = 'Depth map (warm = near)';
    setHighlight(idx);
    try {
      const info = await api(`/api/stage04/depth_info/${frame}`);
      if (idx === state.cameraIdx && info.status === 'ok') {
        $('s04DepthCaption').textContent =
          `Depth map (warm = near) · held-out error ${(info.holdout_rel_err_median * 100).toFixed(1)}% · ${info.n_sparse_points} sparse points`;
      }
    } catch (e) { /* no depth report entry */ }
  }

  async function loadScene() {
    try {
      state.scene = await api('/api/stage04/scene?max_points=30000');
    } catch (err) {
      message(err.message, 'error');
      return;
    }
    const slider = $('s04DepthSlider');
    slider.max = Math.max(0, state.scene.cameras.length - 1);
    state.cameraIdx = 0;
    buildScene(state.scene);
    resizeViewer();
    startLoop();
    showDepth(0);
  }

  // ---------------------------------------------------------------- wiring --------------------
  $('s04ExportBtn').addEventListener('click', exportPackage);
  $('s04ImportBtn').addEventListener('click', () => $('s04ImportInput').click());
  $('s04ImportInput').addEventListener('change', (e) => {
    importResults(e.target.files && e.target.files[0]);
    e.target.value = '';
  });
  $('s04DepthSlider').addEventListener('input', (e) => showDepth(parseInt(e.target.value, 10)));
  ['s04ShowPoints', 's04ShowCameras', 's04ShowPath'].forEach((id) => $(id).addEventListener('change', applyToggles));
  window.addEventListener('resize', () => { if (state.visible) resizeViewer(); });

  function setVisible(on) {
    if (on === state.visible) return;
    state.visible = on;
    section.style.display = on ? 'block' : 'none';
    if (on) {
      refresh().then(() => { resizeViewer(); startLoop(); });
    }
  }

  new MutationObserver(() => setVisible(area.getAttribute('data-active-view') === 'depth-estimation'))
    .observe(area, { attributes: true, attributeFilter: ['data-active-view'] });
})();
