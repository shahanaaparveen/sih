"""
Stage 04 orchestration (camera pose + depth).

The heavy lifting (COLMAP, Depth Anything V2) runs in Google Colab because the local machine has no
GPU. This module owns the two ends of that round trip:

  export_package()  keyframes  -> aero3d_stage04_<project>.zip   (images + manifest + the scripts)
  import_results()  Colab zip  -> validated, parsed, summarised results stored per project
  load_scene()      parsed results -> camera poses + sparse points for the 3D viewer
"""
import json
import os
import shutil
import time
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np

from services.colmap_io import (Model, camera_center, focal_px, frame_index_from_name,
                                horizontal_fov_deg, keyframe_index_from_name, load_model, qvec_to_rotmat)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE04_DIR = os.path.join(BACKEND_DIR, "stage04")
NOTEBOOK_PATH = os.path.join(STAGE04_DIR, "aero3d_stage04_colab.ipynb")
PACKAGE_SCRIPTS = [
    (os.path.join(STAGE04_DIR, "colmap_pipeline.py"), "colmap_pipeline.py"),
    (os.path.join(STAGE04_DIR, "depth_pipeline.py"), "depth_pipeline.py"),
    (os.path.join(BACKEND_DIR, "services", "colmap_io.py"), "colmap_io.py"),
]

MIN_KEYFRAMES = 5
MANY_IMAGES = 400
MAX_RESULT_ZIP_UNCOMPRESSED = 4 * 1024 ** 3
MAX_RESULT_ZIP_FILES = 20000
FORMAT_ID = "aero3d-stage04"

PACKAGE_README = """AERO3D STAGE 04 PACKAGE  (camera pose with COLMAP + depth with Depth Anything V2)

Contents
  images/         the keyframes chosen in Stage 02 (keyframe_*), plus sharp gap-fill frames (fill_*) where
                  Stage 02 left holes in the flight (full resolution)
  manifest.json   which project this is + keyframe list
  colmap_pipeline.py, depth_pipeline.py, colmap_io.py   the code that runs in Colab

How to use
  1. Open aero3d_stage04_colab.ipynb in Google Colab (Runtime > Change runtime type > T4 GPU).
  2. Upload this zip when the notebook asks, then run the cells top to bottom.
  3. Download stage04_results.zip and import it in the Aero3D app (Pose & Depth page).
"""


def _gap_stats(keyframes: List[Dict[str, Any]]) -> Dict[str, Any]:
    idx = sorted(int(k.get("frame_index", 0)) for k in keyframes)
    gaps = np.diff(idx) if len(idx) > 1 else np.array([0])
    return {"median_gap_frames": float(np.median(gaps)), "max_gap_frames": int(gaps.max())}


def plan_gap_fills(keyframes: List[Dict[str, Any]], frames: List[Dict[str, Any]], frames_dir: str) -> List[Dict[str, Any]]:
    """
    Stage 02 picks keyframes by sharpness/novelty, which can leave holes in the flight (for example
    when many consecutive frames are motion-blurred). Structure-from-motion needs every image to
    overlap its neighbours, so a hole splits the reconstruction and loses the images after it.

    For every gap that is abnormally large (> 3x the median gap and > 8 frames) this picks the
    sharpest available video frames inside it as *extra* SfM images. Stage 02's keyframe list
    itself is left untouched.
    """
    idx = sorted({int(k["frame_index"]) for k in keyframes if k.get("frame_index") is not None})
    if len(idx) < 2 or not frames:
        return []
    median = float(np.median(np.diff(idx)))
    max_gap = max(8, int(round(3 * median)))
    target = max(4, int(round(2 * median)))
    sharp = {int(f["frame_index"]): float(f.get("sharpness", 0.0)) for f in frames if f.get("frame_index") is not None}
    taken, fills = set(idx), []
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a
        if gap <= max_gap:
            continue
        n = int(np.ceil(gap / target)) - 1
        seg = gap / (n + 1)
        for j in range(1, n + 1):
            center = a + j * seg
            lo, hi = int(max(a + 1, np.floor(center - seg / 2))), int(min(b - 1, np.ceil(center + seg / 2)))
            candidates = [f for f in range(lo, hi + 1) if f in sharp and f not in taken
                          and os.path.isfile(os.path.join(frames_dir, f"frame_{f:04d}.jpg"))]
            if candidates:
                best = max(candidates, key=lambda f: sharp[f])
                taken.add(best)
                fills.append({"frame_index": best, "sharpness": round(sharp[best], 2)})
    fills.sort(key=lambda f: f["frame_index"])
    return fills[:max(len(keyframes), 1)]


def export_package(results: Dict[str, Any], keyframes_dir: str, frames_dir: str, out_zip: str,
                   project_slug: str) -> Dict[str, Any]:
    keyframes = results.get("keyframes") or []
    present = [k for k in keyframes if k.get("filename") and os.path.isfile(os.path.join(keyframes_dir, k["filename"]))]
    if len(present) < MIN_KEYFRAMES:
        raise ValueError(f"Stage 04 needs at least {MIN_KEYFRAMES} keyframes on disk for this project "
                         f"(found {len(present)}). Process the video first.")

    warnings = []
    if len(present) < len(keyframes):
        warnings.append(f"{len(keyframes) - len(present)} keyframe image(s) are missing on disk and were left out.")
    cuts = [int(c) for c in (results.get("scene_cuts") or [])]
    if cuts:
        warnings.append(f"The video hard-cuts {len(cuts)} time(s) (frames {cuts[:8]}{'...' if len(cuts) > 8 else ''}). "
                        "Structure-from-motion cannot join different shots, so COLMAP will produce roughly one "
                        "reconstruction per shot and only the largest is imported. For a survey, use one "
                        "continuous, unedited flight.")
    if len(present) > MANY_IMAGES:
        warnings.append(f"{len(present)} keyframes is a lot for COLMAP (matching time grows quickly with image "
                        "count). If the Colab run is too slow, re-process the video with the lighter keyframe "
                        "selection (sfm_light).")
    gaps = _gap_stats(present)
    fills = plan_gap_fills(present, results.get("frames") or [], frames_dir)
    if fills:
        warnings.append(f"Stage 02 left gaps of up to {gaps['max_gap_frames']} frames between keyframes "
                        f"(median {gaps['median_gap_frames']:.0f}); {len(fills)} extra sharp frame(s) were added "
                        "so COLMAP keeps a connected reconstruction.")
    elif gaps["max_gap_frames"] > max(10, 5 * gaps["median_gap_frames"]):
        warnings.append(f"Largest gap between consecutive keyframes is {gaps['max_gap_frames']} frames "
                        f"(median {gaps['median_gap_frames']:.0f}) and could not be filled from the stored frames. "
                        "Big gaps break feature matching; COLMAP may split the reconstruction there.")
    for n, f in enumerate(fills):
        f["image"] = f"images/fill_{n:04d}_original_{f['frame_index']:04d}.jpg"

    manifest = {
        "format": FORMAT_ID, "version": 1,
        "project": results.get("filename"), "project_slug": project_slug,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": {k: results.get(k) for k in ("resolution", "width", "height", "fps", "duration", "total_frames")},
        "keyframe_gaps": gaps,
        "fill_frames": fills,
        "scene_cuts": cuts,
        "warnings": warnings,
        "keyframes": [{"keyframe_index": k["keyframe_index"], "frame_index": k.get("frame_index"),
                       "timestamp": k.get("timestamp"), "sharpness": k.get("sharpness"),
                       "image": f"images/{k['filename']}"} for k in present],
    }

    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    tmp = out_zip + ".part"
    with zipfile.ZipFile(tmp, "w") as z:
        for k in present:  # JPEGs are already compressed: store, don't deflate
            z.write(os.path.join(keyframes_dir, k["filename"]), f"images/{k['filename']}", zipfile.ZIP_STORED)
        for f in fills:
            z.write(os.path.join(frames_dir, f"frame_{f['frame_index']:04d}.jpg"), f["image"], zipfile.ZIP_STORED)
        for src, arc in PACKAGE_SCRIPTS:
            z.write(src, arc, zipfile.ZIP_DEFLATED)
        z.writestr("manifest.json", json.dumps(manifest, indent=2), zipfile.ZIP_DEFLATED)
        z.writestr("README.txt", PACKAGE_README, zipfile.ZIP_DEFLATED)
    os.replace(tmp, out_zip)
    manifest["package_bytes"] = os.path.getsize(out_zip)
    return manifest


# ------------------------------------------------------------------------- import -----------

def _safe_extract(zip_path: str, dest: str) -> List[str]:
    root = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        if len(infos) > MAX_RESULT_ZIP_FILES:
            raise ValueError("Results zip has too many files.")
        if sum(i.file_size for i in infos) > MAX_RESULT_ZIP_UNCOMPRESSED:
            raise ValueError("Results zip is too large when extracted.")
        for info in infos:  # zip-slip guard: nothing may land outside dest
            target = os.path.realpath(os.path.join(root, info.filename))
            if os.path.commonpath([root, target]) != root:
                raise ValueError(f"Unsafe path in results zip: {info.filename}")
        z.extractall(root)
        return [i.filename for i in infos]


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def import_results(stage_dir: str, expected_slug: str, expected_keyframes: int, zip_path: str) -> Dict[str, Any]:
    os.makedirs(stage_dir, exist_ok=True)
    staging = os.path.join(stage_dir, "results_incoming")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        try:
            _safe_extract(zip_path, staging)
        except zipfile.BadZipFile:
            raise ValueError("That file is not a valid zip archive.")

        # the notebook may nest everything in one folder; find the model dir wherever it is
        model_dir = next((os.path.dirname(os.path.join(dp, f)) for dp, _, fs in os.walk(staging)
                          for f in fs if f == "images.txt" and os.path.basename(dp) == "model"), None)
        if not model_dir:
            raise ValueError("Results zip has no model/ folder (cameras.txt, images.txt, points3D.txt). "
                             "Was it produced by aero3d_stage04_colab.ipynb?")
        base = os.path.dirname(model_dir)
        for needed in ("cameras.txt", "images.txt", "points3D.txt"):
            if not os.path.isfile(os.path.join(model_dir, needed)):
                raise ValueError(f"model/{needed} is missing from the results zip.")

        manifest = _read_json(os.path.join(base, "manifest.json"))
        warnings: List[str] = []
        if manifest is None:
            warnings.append("Results zip carries no manifest.json, so it could not be matched to this project.")
        elif manifest.get("project_slug") != expected_slug:
            raise ValueError(f"These results belong to project '{manifest.get('project')}', "
                             f"not the active project. Select that project first.")
        elif len(manifest.get("keyframes", [])) != expected_keyframes:
            warnings.append("Keyframe count differs from the current project (it was re-processed after export?).")

        model = load_model(model_dir)
        if not model.images:
            raise ValueError("The reconstruction contains no registered images.")

        final = os.path.join(stage_dir, "results")
        shutil.rmtree(final, ignore_errors=True)
        shutil.move(base, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    summary = summarize(os.path.join(final, "model"), model, manifest, expected_keyframes)
    summary["warnings"] = warnings + summary["warnings"]
    summary["imported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(stage_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _SCENE_CACHE.clear()
    return summary


def summarize(model_dir: str, model: Model, manifest: Optional[Dict], total_keyframes: int) -> Dict[str, Any]:
    results_dir = os.path.dirname(model_dir)
    colmap_rep = _read_json(os.path.join(results_dir, "colmap_report.json")) or {}
    depth_rep = _read_json(os.path.join(results_dir, "depth_report.json")) or {}
    warnings: List[str] = list(colmap_rep.get("warnings", [])) + list(depth_rep.get("warnings", []))

    ordered = sorted(model.images.values(), key=lambda im: (frame_index_from_name(im.name) is None,
                                                            frame_index_from_name(im.name) or 0, im.name))
    registered_idx = [keyframe_index_from_name(im.name) for im in ordered]
    centers = np.array([camera_center(im) for im in ordered])
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.array([0.0])
    med_step = float(np.median(steps)) if steps.size else 0.0
    # Keyframes are not evenly spaced in time, so judge motion per elapsed video frame
    frames = [frame_index_from_name(im.name) for im in ordered]
    if len(steps) > 3 and all(f is not None for f in frames):
        per_frame = steps / np.maximum(np.diff(frames), 1)
    else:
        per_frame = steps
    med_rate = float(np.median(per_frame)) if per_frame.size else 0.0
    jumps = [int(i) for i in np.where(per_frame > 6 * max(med_rate, 1e-9))[0]] if per_frame.size > 3 else []
    if jumps:
        warnings.append(f"{len(jumps)} suspicious camera jump(s) (speed > 6x the median) after video frame(s) "
                        f"{[frames[j] for j in jumps[:5]]}: possible mis-registration.")

    n_reg = len(model.images)
    m_kfs = (manifest or {}).get("keyframes") or []
    fill_total = len((manifest or {}).get("fill_frames") or [])
    kf_indices = {k["keyframe_index"] for k in m_kfs} or set(range(total_keyframes or n_reg))
    kf_total = len(kf_indices)
    reg_kf = {i for i in registered_idx if i is not None}
    kf_reg = len(reg_kf & kf_indices)
    fill_reg = n_reg - len(reg_kf)
    total = kf_total + fill_total
    missing = sorted(kf_indices - reg_kf)

    pts = list(model.points3d.values())
    errs = np.array([p.error for p in pts]) if pts else np.array([0.0])
    tracks = np.array([p.track_length for p in pts]) if pts else np.array([0])
    cam = next(iter(model.cameras.values()))
    mean_err = colmap_rep.get("mean_reprojection_error_px", float(errs.mean()))
    reg_frac = kf_reg / max(1, kf_total)   # keyframes registered in the imported (largest) reconstruction
    n_models = colmap_rep.get("num_models", 1)
    # With several reconstructions the imported one is only part of the story: judge by what registered anywhere
    basis = colmap_rep.get("registered_any_fraction", reg_frac) if n_models > 1 else reg_frac
    cuts = (manifest or {}).get("scene_cuts") or []
    if n_models > 1 and cuts:
        warnings.append(f"{n_models} reconstructions for a video with {len(cuts)} hard cut(s) is expected: each shot "
                        "reconstructs on its own. Only the largest is shown here.")

    if basis < 0.6 or mean_err > 1.5:
        verdict = "poor"
    elif basis >= 0.9 and mean_err <= 0.8 and n_models == 1:
        verdict = "good"
    else:
        verdict = "fair"

    return {
        "format": FORMAT_ID,
        "project": (manifest or {}).get("project"),
        "verdict": verdict,
        "images": {"registered": n_reg, "total": total, "keyframes_registered": kf_reg, "keyframes_total": kf_total,
                   "keyframe_fraction": round(reg_frac, 4), "gap_fill_frames": fill_total,
                   "registered_any_model": colmap_rep.get("num_images_registered_any_model"),
                   "gap_fill_registered": fill_reg, "unregistered_keyframe_indices": missing[:60]},
        "points": {"count": len(pts), "mean_point_error_px": round(float(errs.mean()), 4),
                   "mean_track_length": round(float(tracks.mean()), 2),
                   "median_track_length": float(np.median(tracks))},
        "reprojection_error_px": mean_err,
        "camera": {"model": cam.model, "width": cam.width, "height": cam.height,
                   "focal_px": round(focal_px(cam), 2), "hfov_deg": round(horizontal_fov_deg(cam), 2),
                   "params": [round(float(p), 6) for p in cam.params]},
        "trajectory": {"length_units": round(float(steps.sum()), 4), "median_step_units": round(med_step, 5),
                       "extent_units": [round(float(v), 3) for v in (centers.max(0) - centers.min(0))]},
        "colmap": {k: colmap_rep.get(k) for k in ("colmap_version", "cuda_build", "matcher", "num_models", "models",
                                                  "mean_track_length", "timings_s")},
        "scene_cuts": cuts,
        "depth": {"available": bool(depth_rep), "model": depth_rep.get("model"), "kind": depth_rep.get("kind"),
                  "device": depth_rep.get("device"), "images_with_depth": depth_rep.get("images_with_depth"),
                  "holdout_rel_err_median": depth_rep.get("holdout_rel_err_median"),
                  "holdout_rel_err_p90": depth_rep.get("holdout_rel_err_p90")},
        "units": "COLMAP world units: arbitrary scale and orientation, NOT metres. Metric scale comes from "
                 "georeferencing (Stage 07) and needs real GPS / a known distance.",
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- scene -----------

_SCENE_CACHE: Dict[str, Any] = {}


def load_scene(stage_dir: str, max_points: int = 25000) -> Dict[str, Any]:
    model_dir = os.path.join(stage_dir, "results", "model")
    if not os.path.isfile(os.path.join(model_dir, "images.txt")):
        raise FileNotFoundError("No Stage 04 results imported for this project.")
    key = f"{model_dir}:{os.path.getmtime(os.path.join(model_dir, 'images.txt'))}"
    if key not in _SCENE_CACHE:
        _SCENE_CACHE.clear()
        _SCENE_CACHE[key] = load_model(model_dir)
    model: Model = _SCENE_CACHE[key]

    cameras = []
    for im in sorted(model.images.values(), key=lambda i: frame_index_from_name(i.name) or 0):
        R = qvec_to_rotmat(im.qvec)  # world -> camera; its rows are the camera axes in world coordinates
        cam = model.cameras[im.camera_id]
        cameras.append({
            "name": im.name, "keyframe_index": keyframe_index_from_name(im.name),
            "frame_index": frame_index_from_name(im.name), "is_fill": keyframe_index_from_name(im.name) is None,
            "center": [round(float(v), 5) for v in camera_center(im)],
            "right": [round(float(v), 5) for v in R[0]], "down": [round(float(v), 5) for v in R[1]],
            "forward": [round(float(v), 5) for v in R[2]],
            "hfov_deg": round(horizontal_fov_deg(cam), 3), "aspect": round(cam.width / cam.height, 4),
        })

    pts = [p for p in model.points3d.values() if p.track_length >= 3 and p.error < 2.0] or list(model.points3d.values())
    if len(pts) > max_points:
        keep = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = [pts[i] for i in sorted(keep)]
    xyz = np.array([p.xyz for p in pts]) if pts else np.zeros((0, 3))
    rgb = np.array([p.rgb for p in pts]) if pts else np.zeros((0, 3))
    up = -np.mean([c["down"] for c in cameras], axis=0)
    up = up / (np.linalg.norm(up) or 1.0)
    return {
        "cameras": cameras,
        "points": [round(float(v), 4) for v in xyz.ravel()],
        "colors": [int(v) for v in rgb.ravel()],
        "point_count": len(pts), "total_points": len(model.points3d),
        "up": [round(float(v), 5) for v in up],
    }


def depth_preview_path(stage_dir: str, frame_index: int) -> Optional[str]:
    """Depth preview of a video frame (keyframes and gap-fill frames are both named ..._original_<frame>)."""
    preview_dir = os.path.join(stage_dir, "results", "depth_preview")
    if not os.path.isdir(preview_dir):
        return None
    suffix = f"_original_{frame_index:04d}.jpg"
    for name in sorted(os.listdir(preview_dir)):
        if name.endswith(suffix):
            return os.path.join(preview_dir, name)
    return None


def depth_info(stage_dir: str, frame_index: int) -> Optional[Dict[str, Any]]:
    rep = _read_json(os.path.join(stage_dir, "results", "depth_report.json"))
    if not rep:
        return None
    return next((e for e in rep.get("images", []) if frame_index_from_name(e.get("name", "")) == frame_index), None)
