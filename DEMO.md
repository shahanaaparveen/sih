# Aero3D OnePass — Demo Runbook (SIH 26158)

A repeatable ~5-minute walkthrough of the single-pass drone-video → 3D pipeline, running entirely on
CPU on this machine. Everything shown is produced from the uploaded video; no Colab, no mock numbers.

---

## 0. One-time setup

```bash
py -3.11 -m venv .venv            # or the uv 3.11 interpreter
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
```

- Copy `.env.example` → `.env`. Leave `USE_SUPABASE=false` for a fully offline demo, or set the
  Supabase keys to use Postgres + Storage. The app runs either way (SQLite fallback).
- Put a **short, continuous, textured** clip in `uploads/` — 1080p, 20–45 s, buildings/terrain
  (not sea/sky/water). See the clip spec in `ROADMAP.md §1`.

## 1. Start the app

```bash
.venv\Scripts\python.exe backend/main.py
```

Open **http://localhost:8000**. `/api/health` should report `db_backend: sqlite` (or `supabase`).

---

## 2. The walkthrough

| # | Do this in the UI | What it proves (rubric) |
|---|---|---|
| 1 | **Upload Video** → drop the clip → keyframe mode **SfM (light)** → process | Single-pass ingestion; smart keyframe selection |
| 2 | **Video Processing** / **Trajectory** — show extracted frames, sharpness, 2D path | Frame budget + keyframe overlap (Innovation, Scalability) |
| 3 | **Pose & Depth** → **Run Stage 04 (local)** — watch live progress | Dynamic masking (YOLO) + COLMAP pose + Depth Anything V2, on CPU, no Colab |
| 4 | When done: read the metric tiles (registered, reprojection px, depth error) | **Reconstruction accuracy (30%)** — measured, not claimed |
| 5 | Orbit the 3D viewer (cameras + sparse points + depth per keyframe) | Model completeness, visualization |
| 6 | **Build dense cloud** → toggle **Dense cloud** in the viewer | Dense colored point cloud (**Completeness 20%**) |
| 7 | **Georeference (→ metres)** — show scale + horizontal/vertical RMSE | **Metric accuracy without GCPs** (30%) |
| 8 | Download **PLY / LAS / metric PLY / GeoJSON**; open the PLY in CloudCompare/MeshLab | Standard deliverables (PLY/LAS), GIS track |

## 3. Honest talking points (say these — they are strengths)

- **"Single-pass"** = one continuous flight trajectory, used as a constrained multi-view sequence — not one photo.
- **Depth** is validated on **held-out** sparse points, so the error figure is honest, not a fit residual.
- **GPS is simulated** on these test clips (labelled in the UI). The georeferencing *pipeline* is complete and
  validated; real GPS/RTK (a `frame_index,lat,lon,alt` CSV named like the video in `uploads/`) yields genuine
  metric accuracy. **Do not** claim centimetre accuracy from simulated or ordinary GPS.
- **Speed:** CPU-only. We quote **measured** end-to-end timing (see `benchmark.json` / §4), and treat the SIH
  "<15 min for a 10-min video" figure as a target to benchmark on GPU hardware, not a CPU claim. Near-real-time
  is delivered as a progressive preview; the final textured mesh is a refinement step.
- **Dynamic objects** (people, vehicles, animals) are detected with YOLOv8 segmentation and masked out of
  both COLMAP and the dense cloud, so they don't become false geometry (a named PS challenge).
- What is **not** yet done (be upfront): confidence/coverage map (Phase 8) and textured mesh (Phase 9.1).
  Masking targets moving objects, not sky; some far background may remain (handled by depth far-clipping).

## 4. Benchmark (measured)

Run it yourself:

```bash
.venv\Scripts\python.exe scripts/benchmark.py --clip "uploads/<your_clip>.mp4" --mode sfm_light --budget 120
```

It writes `benchmark.json` and prints a per-stage timing table. Latest measured run:

<!-- BENCHMARK_TABLE -->
**Clip:** college 3840×2160, 23.8 s, 712 frames · **Hardware:** 16-core Intel CPU (no GPU) ·
**Settings:** `sfm_light`, frame budget 120, exhaustive matching

| Stage | Seconds |
|---|---|
| process (frames + keyframes + trajectory) | 23.8 |
| Stage 04 pose + depth (COLMAP 12.6 feat / 50.2 match / 23.2 map + depth) | 124.3 |
| dense cloud (fusion) | 1.8 |
| georeference | 0.3 |
| exports (LAS + GeoJSON) | 0.3 |
| **TOTAL** | **150.4 (~2.5 min)** |

**Result:** 16/16 keyframes registered · verdict **good** · **0.75 px** reprojection · **190,168** dense points ·
16/16 depth maps · georef RMSE **1.23 m** horizontal (simulated GPS). Matching dominates and grows with keyframe
count; `sfm_light` + a short clip keep it fast.

## 5. Reset / troubleshooting

- **Re-run a clip:** just upload again; per-project storage is replaced.
- **COLMAP builds nothing / few images registered:** the clip is too low-texture or motion too fast — use a
  continuous clip over buildings/terrain (the UI warns on <80% registration and on hard cuts).
- **Depth "0 maps":** fixed — alignment auto-detects the model's output convention. If it recurs, the clip has
  too few sparse points; use more keyframes (`sfm` mode).
- **Slow on CPU:** use `sfm_light`, keep the clip short, and a smaller frame budget. Matching dominates; it grows
  with keyframe count.
- **Tests:** `.venv\Scripts\python.exe -m unittest discover -s backend/tests`
