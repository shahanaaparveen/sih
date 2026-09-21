# Single-Pass Drone Video → 3D Model (Prototype UI)

This is a lightweight single-page prototype that visualizes the pipeline steps for the SIH problem statement. It is intended for demonstrations and prototyping only.

Files added:

- `index.html` — main demo page
- `assets/css/styles.css` — styles
- `assets/js/app.js` — populates the pipeline cards and toggles details

How to run

1. Open the project folder and double-click `index.html` in your browser, or serve it with a static server.

Example (Python 3 built-in server):

```bash
python -m http.server 8000
# then open http://localhost:8000 in your browser
```

Next steps (optional):

- Replace summary/details content with richer text or images from the dataset.
- Add thumbnails or embedded video from a recorded flight.
- Hook each stage to demo scripts or small visualizations.

Sample 4K drone videos (royalty-free)

You can download any of these free 4K clips for testing. Example commands below use `curl`, `wget`, or the included scripts in `scripts/`.

- https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4  (sea / seascape, 4K)
- https://cdn.pixabay.com/video/2022/12/09/142199-779684572_large.mp4 (mountains / forest, 4K)
- https://cdn.pixabay.com/video/2017/06/22/10213-222917614_large.mp4 (Grindelwald, Switzerland, 4K)

Download examples:

```bash
# using curl
curl -L -o assets/media/sample_4k.mp4 https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4

# or using wget
wget -O assets/media/sample_4k.mp4 https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4

# or use the included scripts
./scripts/download_sample.sh
powershell -File scripts\download_sample.ps1
```

Notes:
- These files are large (tens to hundreds of MB). Keep an eye on disk space and Git history if you add them to the repository — consider storing large binaries in a release or external storage instead of committing to Git.


## Running the app

```bash
pip install -r backend/requirements.txt
python backend/main.py          # http://localhost:8000  (MongoDB is optional)
```

`server.py` is an older Flask prototype and is not used by the current UI.

## Pipeline status

| Stage | Status |
|---|---|
| 01 Ingestion (metadata) | Working. GPS/IMU/RTK telemetry in the UI is **simulated** and labelled as such. |
| 02 Keyframe extraction | Working. Three selection modes (below) + ORB 2D trajectory. |
| **04 Camera pose (COLMAP) + depth (Depth Anything V2)** | **Working**, runs in Colab (see below). |
| 03 Object purging, 05 Fusion, 06 3D reconstruction, 07 Georeferencing, 08 Validation | UI mock-ups only (marked DEMO). |

## Stage 02: keyframe selection modes

Choose the mode on the Upload page (or `POST /api/process-video?keyframe_mode=...`):

| Mode | What it does |
|---|---|
| `sfm` (default) | Picks the **sharpest frame in each window** so that consecutive keyframes **overlap 88-96%** (overlap measured from the frame-to-frame motion the pipeline already estimates). Accuracy first. |
| `sfm_light` | Same, overlap 65-85%: about 4x fewer images and a much faster COLMAP run, slightly less accurate. |
| `notebook` | The original Colab filter, unchanged: absolute sharpness cutoff (100) + histogram-correlation redundancy filter. |

Why: COLMAP needs each image to overlap its neighbours, and the notebook filter's two tests measure something else (an absolute sharpness number that depends on resolution and scene, and how alike two brightness histograms are). Measured on simulated flights with known camera positions (`scratch/compare_keyframe_modes.py`), the notebook filter lost the last part of every flight and, on a faster flight, kept only 8 frames and registered 5 (38 m of a 116 m path). `sfm` registered every image and covered the whole path on all clips; its orientation error was 38-52% below `sfm_light`, position error 2-41% lower. The presets are those measured points; they have **not yet been checked on real drone footage**.

### Tested on a real clip

A 67 s edited FPV flight (1080p, 2,015 frames, a black fade-in and 5 hard cuts) found two bugs that the simulated flights could not: the first version measured overlap as "how much of the current frame the old frame still covers", which stays ~100% when flying *forward* (the scene zooms in), so it produced 5 keyframes from 2,015 frames; and a black fade-in was picked as keyframes. Both are fixed and covered by regression tests. With `sfm_light` (138 keyframes + 6 gap-fill frames), COLMAP registered 142 of 144 images, 98-100% within every shot. Because the clip is cut from several shots, they form 5 separate reconstructions (structure-from-motion cannot join different shots) and the app imports the largest, so the headline "40% registered" for the largest model alone is misleading; the report now says how many images registered in any model. Hard cuts are detected in Stage 02 (`scene_cuts` in the results, and a warning in the export). For a survey, use one continuous flight.

Practical notes: on CPU that clip took about 20 minutes in COLMAP (4 threads). `colmap_pipeline.py --num-threads N` limits threads (COLMAP decodes one image per thread, so on a machine with little free RAM the default aborts with no message) and `--reuse-sparse` rebuilds `model/` and the report from an existing reconstruction without re-running SfM.

## Stage 04: camera pose and depth (Colab round trip)

No local GPU is needed. The heavy compute runs on a free Colab GPU:

1. **Pose & Depth** page, *Download package*: keyframes (plus sharp *gap-fill* frames where Stage 02 left holes in the flight, because holes split the COLMAP reconstruction) + the scripts.
2. Open `backend/stage04/aero3d_stage04_colab.ipynb` in Colab (T4 GPU), upload the package, run all cells, download `stage04_results.zip`.
3. *Import results* on the same page: the app validates the archive, shows accuracy figures (registered images, reprojection error, held-out depth error), a 3D view of the camera poses and sparse points, and the depth map of every keyframe.

Notes
- Poses/depth are in COLMAP's **arbitrary scale and orientation, not metres**. Metric scale needs georeferencing (real GPS or a known distance).
- Depth Anything V2 predicts relative depth; each map is fitted to COLMAP's sparse points and validated on held-out points.
- `Depth-Anything-V2-Large` is CC-BY-NC-4.0 (non-commercial); `Base`/`Small` are Apache-2.0.
- COLMAP runs in its own virtualenv in the notebook: `pycolmap-cuda12` pins CUDA libraries that conflict with PyTorch.
- `backend/stage04/colmap_pipeline.py` and `depth_pipeline.py` also run locally (CPU) if you install `pycolmap`, `torch`, `transformers`.

Tests: `python -m unittest discover -s backend/tests`. Pose accuracy can be measured against a simulated flight with known camera positions: `scratch/make_synthetic_flight.py` and `scratch/eval_stage04_accuracy.py`.
