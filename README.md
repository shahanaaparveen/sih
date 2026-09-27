# sih3d — Single-Pass Drone Video → Georeferenced 3D Model

Turn one ordinary drone flight — a single pass, no survey-grade mission planning, no mandatory RTK hardware — into a dense, colored, metrically-scaled 3D point cloud and mesh. Built for [SIH 2026].

This README is the current source of truth for the project. Where a claim can be checked against real code or a real run, it is; where something is still aspirational, it's labeled that way. That distinction is a project principle, not a formatting choice — see [Design Principle 0](#0-never-report-a-result-that-has-no-basis) below.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Quick start](#quick-start)
- [Pipeline status — what's actually verified](#pipeline-status--whats-actually-verified)
- [System design](#system-design)
- [Architecture decisions](#architecture-decisions)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Project history](#project-history--superseded-prototype)
- [Sample test footage](#sample-test-footage)

---

## Why this exists

Commercial photogrammetry tools (Pix4D, Metashape, ODM) assume a planned, multi-pass grid flight with 75-85% overlap, often RTK GPS. That's a lot to ask when the flight is improvised, the drone is consumer-grade, or there's no time to plan a survey mission. sih3d is built for the harder, cheaper input case: **one ordinary pass**, standard consumer GPS (or none at all), and whatever a mid-range GPU can process.

The trade-off is explicit and reported honestly, not hidden: a single pass gives you narrower triangulation baselines than a planned grid flight, and standard GPS caps absolute accuracy well below RTK. sih3d does the best correct reconstruction the input actually supports, and says so — see [Known Limitations](#known-limitations).

---

## Quick start

```bash
# 1. Environment (Python 3.10-3.12)
conda create -n sih3d python=3.11 -y && conda activate sih3d
pip install -e .[masking,depth,export,dev]

# 2. System dependencies (not pip packages — check separately)
#    - COLMAP binary on PATH (or set pose.colmap_bin in the config)
#    - ffmpeg on PATH
#    - CUDA available to PyTorch: python -c "import torch; print(torch.cuda.is_available())"

# 3. Sanity check before touching real data
sih3d doctor      # CUDA, COLMAP, transformers, ultralytics, open3d all present?
pytest            # geodesy math, config validation, telemetry parsing

# 4. Validate against known ground truth (catches a broken stage while the right answer is still known)
sih3d synth
sih3d oracle

# 5. Run on a real video
sih3d run --config configs/default.yaml <path-to-video> --run-name <name> [--telemetry <path>]
```

Target hardware: **RTX 3060 (12GB) + 64GB RAM**. A CPU-only fallback config exists at `configs/fast.yaml` for constrained environments.

---

## Pipeline status — what's actually verified

Every stage writes a manifest with real wall-clock timing and is independently resumable — a run can be stopped and restarted without redoing finished stages. This table reflects **verified state as of this writing**, not the target architecture — the two are different, and conflating them is exactly the failure mode this project was rebuilt to eliminate (see [Project History](#project-history--superseded-prototype)).

| Stage | Status | Evidence |
|---|---|---|
| 1 — Frame extraction & keyframing | ✅ **Verified on real footage** | Two real runs on actual 4K drone video: 92 and 244 keyframes selected, real timings logged |
| 2 — Dynamic-object masking (YOLO) | ✅ **Built and working** | — |
| 3 — Structure-from-Motion (COLMAP) | ✅ **Verified on real footage** | Run 1: 100% registration (92/92 images), 0.49px mean reprojection error, 47k sparse points. Run 2: 78% registration (191/244), 129k sparse points, weaker tracks — realistic for a harder flight |
| 4 — Monocular metric depth (DA3) | ❌ **Not yet built** | Critical path — stage 5 cannot be trusted as "real" end to end until this lands |
| 5 — Alignment + TSDF fusion | ⚠️ **Code exists, needs re-verification** | Produced a real mesh in Run 1 (308k points, 689k triangles) — but since stage 4 doesn't exist yet, that output almost certainly did not come from real aligned depth. Do not treat that run's visual quality as representative until stage 4 is built and this stage is re-run against it |
| 6 — Georeferencing | ❌ **Not yet built** | Must report a real RMSE or explicitly "ungeoreferenced" — never a fabricated value (this bit the project once already; see below) |
| 7 — Report + export | ❌ **Not yet built** | `export_glb.py` exists as a standalone script but is not the full report+export stage |
| Website (Aero3D + backend API) | ⚠️ **Partially wired** | Some API endpoints exist; not yet a clean upload → background job → poll → auto-export flow |

**A caught bug, kept here on purpose:** an earlier run showed stage 6 as a green "success" with a 97.8% precision readout — with zero telemetry supplied, meaning there was nothing to measure that number against. It was fabricated. The fix wasn't patching that one number, it was making "never report a result with no basis" a standing rule (below). Recording the bug here, not just the fix, is part of that rule.

---

## System design

![sih3d pipeline diagram — seven stages from drone video to georeferenced model, color-coded by real build status](sih3d_pipeline_diagram.png)

### Data flow

```
video ──▶ [1] Frame extraction & keyframing
              │  budget-aware baseline selection (not fixed time interval)
              ▼
         [2] Dynamic-object masking (YOLO-seg)
              │
              ▼
         [3] Structure-from-Motion (COLMAP)      ──▶ camera poses
              │  sequential matching + bundle adjustment    sparse 3D points
              │  GPS priors folded into BA, if telemetry exists
              ▼
         [4] Monocular metric depth (DA3METRIC-LARGE)
              │  RANSAC-aligned to stage 3's sparse points, per frame
              ▼
         [5] Scale/shift alignment + TSDF fusion (Open3D)  ──▶ textured mesh
              │  grazing-angle rejection, outlier removal,       point cloud
              │  floating-island pruning (depth hallucination)
              ▼
         [6] Georeferencing (Umeyama similarity transform)  ──▶ real-world
              │  honest RMSE, or "ungeoreferenced" — never fabricated   coordinates
              ▼
         [7] Report + export                     ──▶ accuracy report,
                                                        .ply/.obj/.glb/.las
```

### Stage I/O contract

| Input | Required? | Used by | What it narrows |
|---|---|---|---|
| Drone video (1080p/4K) | **Mandatory** | Stage 1 | The only non-optional data source |
| GPS coordinates | **Mandatory**\* | Stage 6 (+ cross-checks Stage 3) | Real-world placement and RMSE |
| Flight metadata (altitude, speed, gimbal, timestamps) | **Mandatory**\* | Stage 1, 3 | SfM initialization scale/orientation prior |
| IMU (accel/gyro) | Optional | Stage 1, 3 | Pose robustness in motion-blurred/low-texture stretches |
| Barometric altitude | Optional | Stage 6 | Vertical accuracy (GPS altitude error is 2-3x horizontal) |
| Camera intrinsics | Optional | Stage 3 | Skips/constrains self-calibration → tighter triangulation |
| RTK/PPK correction | Optional | Stage 6 | The single biggest lever on absolute accuracy — meter-level → centimeter-level |

\* Without GPS/metadata the pipeline still completes and produces a correctly-shaped, correctly-scaled model — stage 6 just reports "ungeoreferenced" instead of placing it in real-world coordinates.

| Output | Format | Produced by |
|---|---|---|
| Dense georeferenced point cloud | `.las`/`.laz` or `.ply` | Stage 5 + 6 |
| Textured 3D mesh | `.obj`/`.glb` | Stage 5, exported Stage 7 |
| Recovered camera trajectory | JSON / COLMAP model | Stage 3 |
| Sparse feature point cloud | `.ply` (intermediate) | Stage 3 |
| DSM / orthomosaic | GeoTIFF | Post-process, optional bonus |
| Accuracy & quality report | HTML/JSON | Stage 7 |

### Evaluation criteria

| Criterion | Target | Notes |
|---|---|---|
| Absolute spatial accuracy | ≤1.0m RMSE without RTK/PPK; cm-level with it | Set by input GPS quality, not software — see [Known Limitations](#known-limitations) |
| Relative (internal) accuracy | Within a few % | Independent of georeferencing — catches an internally-consistent-but-offset model |
| Completeness/coverage | No large unexplained gaps over the flown area | — |
| Robustness | Degrades honestly, never fabricates | No telemetry → "ungeoreferenced," not a fake RMSE |
| Visual/geometric plausibility | No spikes, floating geometry, or implausible extent for the flight altitude | — |
| Processing time | Reasonable for a live demo on target hardware | Wall-clock logged per stage |

---

## Architecture decisions

### 1. Single-pass video, not a planned multi-pass grid flight
**Decision:** design for one ordinary flight pass as the primary input, not a survey-planned mission.
**Why:** removes the two biggest barriers to drone 3D mapping — cost and flight-planning expertise. Most real-world users flying an improvised pass (disaster response, small-site inspection) don't have the equipment or time for a grid mission.
**Consequence, stated plainly:** a single pass has narrower triangulation angles than a planned multi-pass flight. `Mapper.tri_min_angle` is deliberately lowered in stage 3 to keep points the default threshold would reject — a mitigation, not a fix. Absolute geometric accuracy is bounded by this; a future multi-pass option (see [Roadmap](#roadmap)) is the way past it, not more tuning of the single-pass path.

### 2. COLMAP incremental SfM, not visual SLAM
**Decision:** COLMAP's offline incremental bundle adjustment, not a real-time SLAM system (e.g. ORB-SLAM3).
**Why:** no real-time constraint exists here — this is an offline pipeline. Full batch bundle adjustment trades speed for a more accurate final trajectory than SLAM's online, real-time-biased optimization.
**Trade-off accepted:** stage 3 is the slowest stage (minutes, not seconds) — and an implausibly fast stage 3 completion is treated as a bug signal, not a win.

### 3. Monocular neural depth (DA3), not classical dense multi-view stereo
**Decision:** Depth Anything 3 (monocular, per-frame) instead of COLMAP's own dense MVS.
**Why:** classical dense stereo needs wide, converging viewpoints and degrades badly on the narrow baselines and weak/repetitive texture (grass, rooftops) a single nadir pass produces. Monocular depth doesn't depend on cross-frame photometric matching, so it's robust to exactly the geometry this project's input constraint (#1) creates.
**Cost of this choice:** monocular depth has no absolute scale on its own — mitigated by RANSAC-aligning every frame's depth against stage 3's independently, geometrically triangulated sparse points before it's trusted downstream. This is the specific mechanism that makes "use a neural depth model" safe rather than a leap of faith — see the [caught failure mode](#pipeline-status--whats-actually-verified) note above for why that distinction matters in practice, not just in theory.

### 4. TSDF fusion, not naive point-cloud concatenation
**Decision:** volumetric truncated-signed-distance-field fusion (Open3D), not stacking every frame's back-projected points into one cloud.
**Why:** naive concatenation renders every surface N times at slightly different depths — a 2cm-thick wall in reality becomes a fuzzy 20cm slab in the cloud, and no outlier removal fixes it because none of those points is individually wrong. TSDF fusion averages disagreement across frames into one clean surface instead of stacking it.

### 5. Never report a result that has no basis
**This is a cross-cutting principle, not a single decision, and it overrides all the others when they conflict.** No telemetry → stage 6 reports "ungeoreferenced," never a fabricated RMSE. A degraded fallback (e.g. non-neural depth) must be visible in the report, never silent. An implausibly fast stage completion is treated as a bug to investigate, not a milestone. This exists because the project already shipped the opposite once — a fabricated 97.8% precision readout with no telemetry behind it — and the fix was making this a standing rule enforced by the manifest/timing system, not a one-off patch.

### 6. GPS/RTK accuracy ceiling — a constraint, not a decision
Absolute accuracy is capped by whatever GPS quality goes into stage 6 — `geo.ransac_thresh_m` is `4.0` for standard consumer GPS and `0.3` for RTK in the config, directly reflecting this. No amount of pipeline tuning moves this ceiling; only better input (RTK/PPK, or ground control points) does. This is documented here so it's never mistaken for a solvable software gap.

---

## Configuration

Key knobs from `configs/default.yaml` (tuned for an RTX 3060 / 12GB, 1-4 minute single pass at 60-100m AGL):

| Section | Key | Value | Why |
|---|---|---|---|
| `frames` | `candidate_fps` | 4.0 | Probe rate the keyframe selector prunes from |
| `frames` | `max_keyframes` / `min_keyframes` | 600 / 40 | SfM cost budget vs. minimum viable coverage |
| `frames` | `long_edge` | 1600 | Accuracy is bounded by pixels — never dropped for speed |
| `masks` | `model` | `yolo11m-seg.pt` | Segmentation masks, not boxes — preserves static background around moving objects |
| `pose` | `backend` | `colmap` (`da3`/`hybrid` available) | `hybrid` auto-falls-back to DA3 feed-forward pose if COLMAP registers <40% of keyframes |
| `depth` | `model` | `depth-anything/DA3METRIC-LARGE` | Metric-scale monocular depth specialist |
| `fusion` | `voxel_size_m` | 0.08 | ~0.2 for wide terrain, ~0.04 for a single building |
| `geo` | `ransac_thresh_m` | 4.0 (0.3 for RTK) | See [Architecture Decision 6](#6-gpsrtk-accuracy-ceiling--a-constraint-not-a-decision) |

Config uses strict unknown-key rejection — a typo fails loudly at load time instead of silently doing nothing.

---

## Known limitations

- **Absolute accuracy without RTK/GCPs is capped at meter-level.** This is a hardware/input limitation, not a software one — see Architecture Decision 6.
- **Stages 4, 6, 7 are not yet built.** The pipeline cannot currently be called end-to-end verified — see the status table above.
- **The website is not yet fully wired end-to-end.** Currently the most reliable path to a verified result is the CLI.
- **Resolution flexibility (1080p vs. 4K) is not yet validated.** Everything verified so far has been on 4K input; 1080p needs its own explicit test pass, not an assumption that 4K-tuned parameters generalize down.
- **Long/large videos are not yet stress-tested** against the streaming-depth and disk-space handling this would require at scale.

---

## Testing

```bash
pytest                                    # unit tests: geodesy math, config validation, telemetry parsing
sih3d doctor                              # environment sanity (CUDA, COLMAP, deps)
sih3d synth && sih3d oracle               # synthetic ground-truth validation — run before trusting real footage
```

---

## Roadmap

- **Build stages 4, 6, 7** — the critical path to a genuinely end-to-end verified pipeline.
- **Finish website wiring** — background job runner, per-stage status polling, auto-export on completion, frontend hookup to real run IDs (removing the last manual step that caused a synthetic-scene mixup once already).
- **Hybrid depth backend** — classical dense stereo where flight geometry allows it, falling back to DA3 monocular only where it doesn't.
- **Optional second flight pass support** — the direct fix for the single-pass triangulation-angle ceiling, for users who can fly twice.
- **Evaluate [LingBot-Map](https://github.com/Robbyant/lingbot-map)** (Apache-2.0, feed-forward streaming geometry transformer with native aerial-video support and sky masking) as a potential upgrade path for the `pose.backend=da3` fallback specifically — its purpose-built long-sequence drift correction is a more principled answer to a known weakness than the current ad-hoc chunk-and-stitch approach. Not yet integrated or benchmarked against this pipeline's own accuracy — a prototype-first evaluation, not a planned swap.

---

## Project history — superseded prototype

An earlier prototype (Flask `server.py`, since replaced) used a different, more fragmented architecture: an 8-stage pipeline where most stages were UI mock-ups only, a Colab GPU round-trip for pose+depth (pycolmap + Depth Anything **V2**, not the local COLMAP binary + DA3 **V3** this project now uses), and three named keyframe-selection modes (`sfm`/`sfm_light`/`notebook`) tuned and regression-tested against a real 67-second FPV clip. That work surfaced real, useful findings — a keyframe-overlap measurement bug, false keyframes from a black fade-in, the fact that hard cuts split a COLMAP reconstruction into disconnected sub-models — and some of its testing patterns (a synthetic-flight generator with known ground truth, `scratch/eval_stage04_accuracy.py`) informed this project's current `sih3d synth`/`sih3d oracle` validation approach. It is no longer the current architecture; this README describes what replaced it.

---

## Sample test footage

Free 4K clips for testing (royalty-free, via Pixabay):

```bash
curl -L -o sample_4k.mp4 https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4   # sea/seascape
curl -L -o sample_4k.mp4 https://cdn.pixabay.com/video/2022/12/09/142199-779684572_large.mp4   # mountains/forest
curl -L -o sample_4k.mp4 https://cdn.pixabay.com/video/2017/06/22/10213-222917614_large.mp4    # Grindelwald, Switzerland
```

These files are large (tens to hundreds of MB) — don't commit them to Git; store in a release or external storage instead.
