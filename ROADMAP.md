# Aero3D OnePass — Complete Build Roadmap (Prototype → Product)

**Problem statement:** SIH 26158 — Single-Pass Drone Video to Accurate 3D Model Generation (NTRO).
**This file** is the master TODO. It splits the remaining work into small, ordered, *individually testable* stages. Do them **in order**. Do **not** start a stage until the previous stage's **"Done when"** check passes. That is how we avoid errors.

---

## 0. Chosen configuration (from your answers)

| Decision | Choice | Consequence baked into this plan |
|---|---|---|
| Compute | **CPU only** | No Colab round-trip. Everything runs locally on CPU. Heavy stages are slow → we design for **short demo clips** and a **progressive preview**, not real-time. |
| Primary 3D output | **Dense point cloud (guaranteed) + mesh (stretch)** | Dense colored `.ply`/`.las` is the committed deliverable. Poisson mesh `.obj`/`.glb` is an optional stage that can be skipped if unstable. |
| Data layer | **Supabase** (Postgres + Auth + Storage) | MongoDB is removed. Projects, telemetry, job records → Postgres. Videos/results → Storage buckets. A **SQLite fallback** is kept so the app still runs if Supabase is unreachable. |
| Deployment | **Single-user local demo** | Minimal auth (one login). We still fix upload/CORS/static-mount security, but skip full multi-tenant isolation. |

---

## 1. ⚠️ Reality check — read before doing anything

CPU-only changes what "complete" means. Be honest in the demo and in front of judges:

- **A 10-minute 4K clip will NOT finish in 15 minutes on CPU.** The SIH `<15 min` figure is a *target to benchmark against*, not a claim to make. Your own README already measured ~20 min of COLMAP on a 67 s clip.
- **Demo clip spec (use this for every test and the final demo):**
  - **1080p** (downscaled from 4K is fine), **20–45 seconds**, **one continuous shot** (no cuts/fades).
  - Scene with **texture and structure** (buildings, terrain, roads). **Avoid** sea, sky, flat grass, water — they have no features and COLMAP will fail.
  - Forward or orbit motion with visible parallax.
- **Positioning line for judges (from the capsule):** *"Near-real-time means progressive scene awareness first; the final high-quality textured mesh is refined afterward. We benchmark end-to-end timing on real hardware and never claim real-time we have not measured."*
- Every impressive number shown in the UI must come from a **real run** or be labelled **DEMO**. No invented GSD/vertex/overlap figures.

---

## 2. Corrected tech stack (what you asked)

| Layer | Keep / Change / Add | Choice | Why |
|---|---|---|---|
| Web server | Keep | **FastAPI + Uvicorn** | Correct. Async, handles background jobs. |
| Frontend | Keep | **index.html + vanilla JS + Three.js** | Already strong; wire it to real data. |
| Video / CV | Keep | **OpenCV + NumPy** | Correct. |
| Keyframes | Keep | **Existing `sfm` selector** | Genuinely good — do not rewrite. |
| Camera pose (SfM) | Keep, run **local CPU** | **pycolmap** | Has Windows CPU wheels. Drop the Colab round-trip. |
| AI depth | Keep, run **local CPU** | **Depth Anything V2 – Small/Base** (Apache-2.0) | `Large` is CC-BY-NC (non-commercial) — do not ship it. Small/Base run on CPU. |
| Dynamic masking | **Add** | **Ultralytics YOLOv8n** (CPU) | Masks vehicles/people/animals before fusion. Named in the PS. |
| Dense + mesh | **Add** | **Open3D** | Depth fusion → dense cloud → Poisson mesh → texture. |
| Georeferencing | **Add (wire existing math)** | `similarity_align` (Umeyama) in `colmap_io.py` | Already written + unit-tested, currently unused. Aligns COLMAP → GPS → metres. |
| Point cloud export | **Add** | **laspy** (LAS), Open3D (PLY) | Real deliverables replacing the dead demo buttons. |
| Mesh export | **Add** | **trimesh / pygltflib** | OBJ + GLB. |
| Geo export | **Add (honest subset)** | **GeoJSON/KML** track + bbox; GeoTIFF orthomosaic = **stretch** | Full orthomosaic needs GDAL/rasterio (painful on Windows) and true orthorectification. Ship GeoJSON/KML first. |
| Database | **Change** | **Supabase Postgres** (+ SQLite fallback) | Removes MongoDB. Adds auth + storage in one service. |
| Auth | **Add (minimal)** | **Supabase Auth**, single user | Closes the "no auth" gap cheaply. |
| Object/file storage | **Change** | **Supabase Storage buckets** | Replaces serving the whole `/storage` dir statically. |
| Legacy | **Remove** | `server.py` (Flask), unused `frame_extractor.py` / `keyframe_selector.py` / `trajectory.py` imports | Dead code = future bugs. |

---

## 3. Target repo layout (after the roadmap is complete)

```
SIH/
├── ROADMAP.md                     ← this file
├── README.md                      ← current-state doc (kept)
├── .env.example                   ← Supabase keys template (NEW)
├── index.html                     ← frontend (wired to real data)
├── assets/                        ← css/js/media
├── backend/
│   ├── main.py                    ← FastAPI app (Supabase, not Mongo)
│   ├── requirements.txt           ← pinned, CPU wheels
│   ├── config.py                  ← env + settings + limits (NEW)
│   ├── db/
│   │   ├── supabase_client.py     ← Supabase wrapper (NEW)
│   │   └── sqlite_fallback.py     ← offline fallback (NEW)
│   ├── services/
│   │   ├── video_processor.py     ← Stage 02 (keep, add frame-budget)
│   │   ├── keyframe_sfm.py        ← keep
│   │   ├── colmap_io.py           ← keep (+ wire similarity_align)
│   │   ├── masking.py             ← Stage 03 YOLO masks (NEW)
│   │   ├── sfm_local.py           ← Stage 04a COLMAP local runner (NEW)
│   │   ├── depth_local.py         ← Stage 04b depth local runner (NEW)
│   │   ├── fusion.py              ← Stage 05 dense cloud (NEW)
│   │   ├── georef.py              ← Stage 07 GPS align + RMSE (NEW)
│   │   ├── meshing.py             ← Stage 06 Open3D mesh (NEW)
│   │   ├── confidence.py          ← Stage 08 coverage/confidence (NEW)
│   │   └── exporter.py            ← Stage 09 PLY/LAS/OBJ/GLB/GeoJSON (NEW)
│   ├── jobs/
│   │   └── pipeline_runner.py     ← background job orchestrator (NEW)
│   ├── storage/                   ← local work dir (gitignored)
│   └── tests/                     ← unittest, grows each stage
└── scripts/                       ← helpers
```

---

## 4. What YOU must provide / place in the folder

Tick these off first — several stages are blocked without them.

- [ ] **Supabase project.** Either (a) tell me to create/link it via the Supabase plugin here, or (b) create one at supabase.com and give me: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_PASSWORD`. I'll never print the service key back.
- [ ] **A demo drone video** matching the clip spec in §1. Place it in `uploads/`. If you don't have one yet, use a Pixabay 4K clip from the current README, but pick a **city/terrain** clip, not sea/sky.
- [ ] **(Optional) A real telemetry/GPS file** if your dataset has one (CSV/SRT/GPX). If not, we simulate GPS and clearly label it until the SIH dataset arrives.
- [ ] **Confirm Python version.** Recommended **Python 3.11 (64-bit)** on Windows — pycolmap, Open3D, and Torch CPU all have 3.11 wheels. (3.12 sometimes lacks a pycolmap wheel.)
- [ ] **Disk space:** keep **~15 GB free**. CPU SfM + depth + dense fusion writes a lot.

---

## 5. Stage conventions

Every micro-stage below has the same shape:

- **Goal** — one sentence.
- **Do** — the concrete change.
- **Verify** — the exact command/click that proves it works.
- **Done when** — the pass condition. **Do not proceed until this is green.**

Commit after every micro-stage with the stage id in the message (e.g. `git commit -m "P1.2: env + config"`). If a stage fails, you can always `git reset --hard` back to the last green stage — that is the "no error would occur" safety net.

---

# PHASE 0 — Foundation & safety net

### 0.1 Freeze a known-good baseline
- **Goal:** be able to roll back at any time.
- **Do:** `git add -A && git commit -m "P0.1: baseline before rebuild"` then `git tag baseline-pre-rebuild`.
- **Verify:** `git tag` lists `baseline-pre-rebuild`.
- **Done when:** tag exists and `git status` is clean.

### 0.2 Create the Python 3.11 virtual environment
- **Goal:** isolated, reproducible deps.
- **Do:** `py -3.11 -m venv .venv` then activate: `.venv\Scripts\Activate.ps1`.
- **Verify:** `python --version` → `3.11.x`.
- **Done when:** prompt shows `(.venv)` and version is 3.11.

### 0.3 Pin the CPU dependency set
- **Goal:** one install that works on Windows CPU.
- **Do:** rewrite `backend/requirements.txt` (I will generate exact pins). Torch CPU comes from the CPU index URL. Install in this order to avoid resolver conflicts:
  1. `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  2. `pip install -r backend/requirements.txt` (fastapi, uvicorn, python-multipart, opencv-python, numpy, transformers, pycolmap, open3d, ultralytics, laspy, trimesh, pygltflib, supabase, python-dotenv)
- **Verify:** `python -c "import cv2, numpy, pycolmap, open3d, torch, transformers, ultralytics, laspy, trimesh, supabase; print('ok')"`.
- **Done when:** prints `ok` with no ImportError. (If `pycolmap` fails to import, stop — we fix the wheel before anything else.)

### 0.4 Smoke-test pycolmap on CPU
- **Goal:** prove the hardest dependency actually runs, before building on it.
- **Do:** `python -c "import pycolmap; print(pycolmap.__version__, 'cuda:', getattr(pycolmap,'has_cuda',False))"`.
- **Verify:** prints a version and `cuda: False`.
- **Done when:** no crash. (This is the #1 thing that breaks on Windows; catching it now saves days.)

---

# PHASE 1 — Supabase data layer (replaces MongoDB)

### 1.1 Provision the Supabase project
- **Goal:** a live Postgres + Storage backend.
- **Do:** create/link the project (via the plugin here, or you supply keys). Copy `.env.example` → `.env` and fill keys. Ensure `.env` is gitignored.
- **Verify:** `python -c "import os,dotenv;dotenv.load_dotenv();print(bool(os.getenv('SUPABASE_URL')))"` → `True`.
- **Done when:** keys load and `.env` is NOT tracked by git.

### 1.2 Create the database schema
- **Goal:** tables for projects, telemetry, jobs, results.
- **Do:** apply a migration: `projects`, `telemetry`, `pipeline_jobs`, `stage_results` (with a `user_id`, timestamps, and a `metrics jsonb`). RLS on, single-user policy.
- **Verify:** list tables via the Supabase plugin; all four exist.
- **Done when:** tables + RLS policies present.

### 1.3 Create Storage buckets
- **Goal:** hold videos and result artifacts off the web root.
- **Do:** buckets `videos` (private), `results` (private). No public listing.
- **Verify:** buckets exist; anonymous fetch of a known path is denied.
- **Done when:** both buckets exist and are private.

### 1.4 Supabase client wrapper + SQLite fallback
- **Goal:** one module the backend calls; app still works offline.
- **Do:** `backend/db/supabase_client.py` (insert/select/upload) and `backend/db/sqlite_fallback.py` (same interface, local file). A `USE_SUPABASE` flag picks one; if Supabase errors on startup, auto-fallback to SQLite and log a warning.
- **Verify:** unit test `tests/test_db.py` inserts + reads a project row under both backends.
- **Done when:** test passes with Supabase on AND with it forced off.

### 1.5 Cut MongoDB out of `main.py`
- **Goal:** remove the dead DB and its endpoints' Mongo calls.
- **Do:** replace `get_db()`/`serialize_doc()` and the `/api/telemetry`, `/api/pipeline/*` bodies with the wrapper. Delete `pymongo`/`bson` imports.
- **Verify:** `grep -rn "pymongo\|mongo\|bson" backend/main.py` → no matches; server starts.
- **Done when:** app boots, `/api/health` returns `db: supabase|sqlite`, no Mongo references remain.

---

# PHASE 2 — Security & hygiene (cheap, do before feature work)

### 2.1 Central config + limits
- **Goal:** one place for CORS origins, upload caps, model choices.
- **Do:** `backend/config.py` with `MAX_UPLOAD_MB=512`, `ALLOWED_VIDEO_EXT`, `CORS_ORIGINS=["http://localhost:8000"]`, `DEPTH_MODEL="depth-anything/Depth-Anything-V2-Small-hf"`.
- **Verify:** `python -c "from backend.config import settings; print(settings.MAX_UPLOAD_MB)"`.
- **Done when:** config imports and values are read by `main.py`.

### 2.2 Fix CORS
- **Goal:** valid, safe CORS.
- **Do:** replace `allow_origins=["*"] + allow_credentials=True` with explicit `CORS_ORIGINS` and `allow_credentials=True`.
- **Verify:** browser demo still works; a request from a random origin is blocked.
- **Done when:** localhost works, others don't.

### 2.3 Upload guards
- **Goal:** no disk-exhaustion, no junk files.
- **Do:** in `/api/process-video` and `/api/stage04/import`, enforce size cap (stream + count bytes, abort over cap), extension allowlist, and route saves through **one** sanitizer (`project_slug`).
- **Verify:** uploading a 1 GB file returns `413`; uploading `.exe` returns `400`; a filename `..\evil.mp4` is stored as a safe slug.
- **Done when:** all three behave as above.

### 2.4 Stop serving the storage tree; sign URLs instead
- **Goal:** no cross-project/raw-video leakage.
- **Do:** remove `app.mount("/storage")` and `/uploads` static mounts. Serve artifacts via short-lived Supabase **signed URLs** (or an authenticated endpoint in SQLite mode).
- **Verify:** direct GET of another project's frame path → `404/403`; the viewer still loads its own data via signed URL.
- **Done when:** no static mount exposes `storage/` or `uploads/`.

### 2.5 Remove dead code
- **Goal:** kill drift.
- **Do:** delete `server.py`; remove the unused `extract_all_frames` / `calculate_sharpness` / `select_keyframes` / `calculate_camera_trajectory` imports and modules (or fold logic if still referenced — verify with grep first).
- **Verify:** `python -m unittest discover -s backend/tests` still green; app boots.
- **Done when:** dead files gone, tests pass.

### 2.6 Remove fabricated counts
- **Goal:** integrity — no invented stats.
- **Do:** in the projects list / `select_project` fallbacks, set unknown `keyframes`/`trajectory_points` to `null` + `status:"NOT_PROCESSED"` instead of `total_frames`. In the UI, render `—` for nulls.
- **Verify:** select an unprocessed video → UI shows `—`, not fake numbers.
- **Done when:** no code path sets keyframes = total_frames.

---

# PHASE 3 — Pipeline: ingestion & frame budget (fix the speed killer)

### 3.1 Frame budget instead of all-frames-to-disk
- **Goal:** stop writing 18k full-res JPEGs.
- **Do:** in `video_processor._process_frames`, add `MAX_FRAMES_ON_DISK` (e.g. 1500) and a stride so long clips are sampled; keep sharpness/trajectory on the sampled set; decode others on demand.
- **Verify:** process a 45 s 1080p clip → `< 1500` JPEGs written; peak disk under a few hundred MB.
- **Done when:** frame count bounded, keyframe quality unchanged on the demo clip.

### 3.2 Telemetry ingestion (real if present, simulated + labelled if not)
- **Goal:** GPS ready for georeferencing.
- **Do:** parse optional GPX/SRT/CSV → per-frame lat/lon/alt; if absent, synthesize a plausible track and tag every record `source:"SIMULATED"`.
- **Verify:** `/api/telemetry` returns per-frame GPS with a truthful `source` field.
- **Done when:** both real-file and no-file paths work and are labelled.

---

# PHASE 4 — Dynamic object masking (Stage 03, new)

### 4.1 YOLOv8n mask service
- **Goal:** masks for people/vehicles/animals per keyframe.
- **Do:** `services/masking.py` — run YOLOv8n (CPU) on each keyframe, output a binary mask PNG (dynamic pixels = ignore).
- **Verify:** on a keyframe containing a car, the mask covers the car.
- **Done when:** masks generated for every keyframe; time logged.

### 4.2 Feed masks into SfM
- **Goal:** dynamic pixels don't create false geometry.
- **Do:** pass masks to COLMAP feature extraction (mask images), or filter masked keypoints before matching.
- **Verify:** compare sparse cloud with/without masking on a clip with a moving car → fewer stray points on the car.
- **Done when:** masked run registers ≥ unmasked and drops moving-object points.

---

# PHASE 5 — Local SfM + depth (Stage 04, no Colab)

### 5.1 Local COLMAP runner as a background job
- **Goal:** run `colmap_pipeline.py` in-process on CPU without freezing the API.
- **Do:** `services/sfm_local.py` + `jobs/pipeline_runner.py` (thread/subprocess) writing progress to the job record; force `--device cpu`, cap `--num-threads` to avoid RAM aborts, `--max-image-size 1600` for CPU speed.
- **Verify:** POST a run on the demo clip → job progresses → `model/` produced; `/api/stage04/status` shows registered images.
- **Done when:** sparse model builds on CPU from the demo clip; API stayed responsive.

### 5.2 Local depth runner
- **Goal:** aligned depth maps on CPU with the Apache-2.0 model.
- **Do:** `services/depth_local.py` calling `depth_pipeline.py` with `Depth-Anything-V2-Small-hf`, `--device cpu`, smaller `--input-size` (e.g. 518) for speed.
- **Verify:** `depth_report.json` written; median held-out error printed and stored.
- **Done when:** every registered image (or its skip reason) has a depth entry.

### 5.3 Retire the Colab UI path
- **Goal:** one-click local run, no export/import round-trip.
- **Do:** replace the Pose & Depth page's "download package / import results" with "Run Stage 04 (local)" + progress; keep the Colab notebook in-repo as an optional advanced path only.
- **Verify:** clicking Run produces poses + depth without any manual upload.
- **Done when:** the round-trip is no longer required for a demo.

---

# PHASE 6 — Dense point cloud (Stage 05, the guaranteed deliverable)

### 6.1 Depth → per-image colored points
- **Goal:** back-project each aligned depth map into world points.
- **Do:** `services/fusion.py` — for each image, unproject depth using COLMAP intrinsics/pose, attach RGB, drop masked (dynamic) and low-confidence pixels.
- **Verify:** per-image point count reasonable; points land near the sparse cloud.
- **Done when:** at least the demo clip yields per-image clouds aligned to the sparse model.

### 6.2 Fuse + clean into one dense cloud
- **Goal:** a single dense colored cloud.
- **Do:** merge, voxel-downsample (Open3D), statistical outlier removal.
- **Verify:** open the fused `.ply` in the viewer — recognisable scene, no dynamic-object smear.
- **Done when:** one clean dense `.ply` exists for the demo clip.

---

# PHASE 7 — Georeferencing → metres (Stage 07, wire existing math)

### 7.1 Align COLMAP world → GPS
- **Goal:** turn arbitrary units into metric ENU.
- **Do:** `services/georef.py` — take COLMAP camera centres + per-frame GPS (ENU), run `similarity_align` (already in `colmap_io.py`) to get scale+rotation+translation; apply to cloud/mesh/poses.
- **Verify:** median camera-position residual reported; scale is physically sane (metres).
- **Done when:** a metric cloud + a transform are produced; residuals stored.

### 7.2 Accuracy report (honest, GCP-free)
- **Goal:** the 30%-weighted "reconstruction accuracy" evidence.
- **Do:** report horizontal/vertical RMSE vs held-out GPS points, plus a **confidence caveat** when GPS is simulated ("metric scale is indicative until real GPS/RTK is supplied").
- **Verify:** `/api/stage07/report` returns RMSE numbers + the caveat flag.
- **Done when:** report renders in the UI with the caveat when applicable.

---

# PHASE 8 — Confidence & coverage map (Stage 08, the differentiator)

### 8.1 Per-region confidence
- **Goal:** show measured vs weakly-constrained areas.
- **Do:** `services/confidence.py` — combine reprojection error, track length, held-out depth error, keyframe overlap, view count into a 0–1 confidence per point/region.
- **Verify:** low-texture / single-view regions score low; well-seen facades score high.
- **Done when:** the viewer can color the cloud by confidence.

### 8.2 Coverage-gap map
- **Goal:** turn the single-pass limitation into an insight.
- **Do:** flag areas seen by too few views; expose as an overlay + a summary %.
- **Verify:** the side of a building never filmed shows as a gap.
- **Done when:** overlay + "well-observed %" shown.

---

# PHASE 9 — Mesh (stretch), exports & viewer wiring

### 9.1 Poisson mesh + texture (STRETCH — skippable)
- **Goal:** optional `.obj`/`.glb` textured mesh.
- **Do:** `services/meshing.py` — Open3D normal estimation → Poisson → crop to density → vertex colors (texture bake if time permits). Wrap in try/except: on failure, mark mesh "unavailable" and keep the dense cloud as the output.
- **Verify:** mesh opens in a viewer; failure path degrades gracefully to point cloud.
- **Done when:** mesh produced on the demo clip OR cleanly skipped without breaking the run.

### 9.2 Real exports (replace the demo buttons)
- **Goal:** downloads that are actual files.
- **Do:** `services/exporter.py` — `.ply` (Open3D), `.las` (laspy), `.obj`+`.glb` (trimesh/pygltflib) when mesh exists, `.geojson`/`.kml` camera track + footprint. Upload to the `results` bucket; UI links to signed URLs. **GeoTIFF orthomosaic stays a labelled stretch goal.**
- **Verify:** each button downloads a file that opens in CloudCompare/Blender/QGIS.
- **Done when:** PLY + LAS + GeoJSON always work; OBJ/GLB work when mesh exists.

### 9.3 Wire the Three.js viewer to real data
- **Goal:** the 3D view shows the real cloud/mesh + real metrics.
- **Do:** load the fused cloud/mesh from signed URLs; drive the metric tiles (points, RMSE, coverage %, confidence) from real reports; remove hardcoded `1.4 cm / 485K / 99.4%` or gate them behind an unmistakable DEMO badge.
- **Verify:** numbers in the UI match the JSON reports for the demo clip.
- **Done when:** no fabricated metric remains on a "processed" project.

### 9.4 Basic measurement tool
- **Goal:** the "suitable for measurement" requirement + 5% UI score.
- **Do:** point-to-point distance in the viewer using metric coordinates from Phase 7.
- **Verify:** measuring a known-length object reads close to reality (report the error honestly).
- **Done when:** distance measurement works on the metric cloud.

---

# PHASE 10 — Acceptance, benchmark & demo runbook

### 10.1 End-to-end benchmark
- **Goal:** truthful timing for the "processing speed" criterion.
- **Do:** run the full pipeline on the demo clip; record wall-clock per stage; write `benchmark.json`.
- **Verify:** total time + per-stage breakdown captured.
- **Done when:** you can state real numbers ("X min for a Y-second 1080p clip on CPU").

### 10.2 Demo script
- **Goal:** a repeatable 5-minute judge demo.
- **Do:** document: upload demo clip → keyframes → masks → pose/depth (progress) → dense cloud → georef RMSE → confidence overlay → export → measure.
- **Verify:** a fresh person can follow it start to finish.
- **Done when:** the script runs clean twice in a row.

---

## 11. Acceptance matrix (mapped to the SIH rubric)

| Criterion (weight) | Satisfied by | Green when |
|---|---|---|
| Reconstruction accuracy (30%) | P5–P7 | Metric cloud + RMSE report exist |
| Model completeness (20%) | P6, P8 | Dense cloud + coverage % shown |
| Processing speed (20%) | P3.1, P10.1 | Real end-to-end benchmark recorded |
| Innovation (15%) | P2.6, P4, P8 | Confidence + honest single-pass framing demoed |
| Scalability (10%) | P3.1, P5.1 | Bounded frames + background job |
| User interface (5%) | P9.3, P9.4 | Real metrics + measurement in viewer |

---

## 12. Risk register

| Risk | Likelihood | Mitigation (already in plan) |
|---|---|---|
| pycolmap won't install/run on Windows CPU | Med | P0.4 gate before any dependent work |
| CPU too slow for judges | High | Short clip spec (§1); progressive preview; benchmark honestly |
| COLMAP fails on low-texture clip | Med | Clip spec forbids sea/sky; warn in UI on <80% registration |
| Supabase offline during demo | Low | SQLite fallback (P1.4) |
| Mesh (Poisson) unstable | Med | Mesh is a stretch; dense cloud is the committed output (P9.1 graceful skip) |
| GeoTIFF/GDAL pain on Windows | High | Ship GeoJSON/KML; GeoTIFF stays labelled stretch |
| Simulated GPS mistaken for real metric truth | Med | `source:"SIMULATED"` labels + caveat in accuracy report |

---

## 13. Execution order (one line)

**P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10.**
Never skip a "Done when". Commit per micro-stage. The dense point cloud (through P6+P7) is the minimum complete prototype; P8–P9 make it competitive; the mesh is the stretch.
