import os
import sys
import json
import shutil
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List

# Windows console UTF-8 fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Path Configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import settings
from db.store import get_store
from services.video_processor import (process_video_pipeline, get_video_metadata, get_current_progress, project_slug,
                                      KEYFRAME_MODES, DEFAULT_KEYFRAME_MODE)
from services import stage04 as stage04_service

STORAGE_DIR = os.path.join(BACKEND_DIR, "storage")
VIDEOS_DIR = os.path.join(STORAGE_DIR, "videos")
FRAMES_DIR = os.path.join(STORAGE_DIR, "frames")
KEYFRAMES_DIR = os.path.join(STORAGE_DIR, "keyframes")
TRAJECTORIES_DIR = os.path.join(STORAGE_DIR, "trajectories")
STAGE04_STORAGE = os.path.join(STORAGE_DIR, "stage04")

for d in [VIDEOS_DIR, FRAMES_DIR, KEYFRAMES_DIR, TRAJECTORIES_DIR, STAGE04_STORAGE]:
    os.makedirs(d, exist_ok=True)

STORAGE_CONFIG = {
    "videos": VIDEOS_DIR,
    "frames": FRAMES_DIR,
    "keyframes": KEYFRAMES_DIR,
    "trajectories": TRAJECTORIES_DIR
}

# FastAPI App
app = FastAPI(
    title="Aero3D Video Processing API",
    description="FastAPI Backend for single-pass drone video processing, frame extraction, keyframe selection, and camera trajectory",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Document store (SQLite by default, Supabase when configured). See db/store.py.
DB_NAME = "aero3d_db"

# In-memory and disk cache for current video pipeline state
CACHE_FILE = os.path.join(STORAGE_DIR, "latest_pipeline_results.json")
latest_results: Optional[Dict[str, Any]] = None

def get_project_cache_path(filename: str) -> str:
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    return os.path.join(STORAGE_DIR, f"results_{safe_name}.json")

def load_cached_results():
    global latest_results
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                latest_results = json.load(f)
                fname = latest_results.get("filename")
                print(f"[CACHE] Loaded latest results for {fname}")
                if fname:
                    p_path = get_project_cache_path(fname)
                    if not os.path.exists(p_path):
                        with open(p_path, "w", encoding="utf-8") as pf:
                            json.dump(latest_results, pf, indent=2)
        except Exception as e:
            print(f"[CACHE ERROR] {e}")

load_cached_results()

def save_cached_results(data: Dict[str, Any]):
    global latest_results
    latest_results = data
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        fname = data.get("filename")
        if fname:
            p_path = get_project_cache_path(fname)
            with open(p_path, "w", encoding="utf-8") as pf:
                json.dump(data, pf, indent=2)
    except Exception as e:
        print(f"[CACHE ERROR] {e}")


# ==========================================
# ACTIVE PROJECT HELPERS (frames/keyframes are stored per project)
# ==========================================

# Only one video may be processed at a time (pipeline progress is a single global tracker)
PIPELINE_LOCK = threading.Lock()

def _active_slug() -> Optional[str]:
    if latest_results:
        if latest_results.get("project_slug"):
            return latest_results["project_slug"]
        if latest_results.get("filename"):
            return project_slug(latest_results["filename"])
    return None

def active_frames_dir() -> str:
    slug = _active_slug()
    per_project = os.path.join(FRAMES_DIR, slug) if slug else None
    # Results cached before per-project storage keep their frames directly in FRAMES_DIR
    return per_project if per_project and os.path.isdir(per_project) else FRAMES_DIR

def active_keyframes_dir() -> str:
    slug = _active_slug()
    per_project = os.path.join(KEYFRAMES_DIR, slug) if slug else None
    return per_project if per_project and os.path.isdir(per_project) else KEYFRAMES_DIR

def active_video_path() -> Optional[str]:
    if not (latest_results and latest_results.get("filename")):
        return None
    safe_name = "".join(c for c in latest_results["filename"] if c.isalnum() or c in "._- ")
    for base in (VIDEOS_DIR, os.path.join(PROJECT_ROOT, "uploads")):
        candidate = os.path.join(base, safe_name)
        if os.path.isfile(candidate):
            return candidate
    return None

def load_frame(frame_index: int) -> Optional[np.ndarray]:
    """
    Loads a frame of the active project: from its stored JPEG, else straight from the source video.
    """
    stored = os.path.join(active_frames_dir(), f"frame_{frame_index:04d}.jpg")
    if os.path.exists(stored):
        img = cv2.imread(stored)
        if img is not None:
            return img
    video = active_video_path()
    if video:
        cap = cv2.VideoCapture(video)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, img = cap.read()
            return img if ok else None
        finally:
            cap.release()
    return None

# ==========================================
# 1. VIDEO PROCESSING API
# ==========================================

@app.post("/api/process-video")
def process_video_endpoint(
    video: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    sharpness_threshold: float = 100.0,
    similarity_threshold: float = 0.95,
    keyframe_mode: str = DEFAULT_KEYFRAME_MODE
):
    """
    Accepts video upload (multipart/form-data) and runs the complete Colab pipeline:
    - Extracts EVERY frame
    - Calculates metadata
    - Calculates Laplacian sharpness score for every frame
    - Filters sharp frames
    - Performs visual redundancy filtering & saves keyframes
    - Calculates camera trajectory using ALL extracted frames
    - Returns structured results
    """
    # Plain `def` (not `async def`): FastAPI runs it in a worker thread, so the long OpenCV
    # pass below does not freeze the event loop and /api/progress keeps answering.
    upload_file = video or file
    if not upload_file or not upload_file.filename:
        raise HTTPException(status_code=400, detail="No video file provided in multipart/form-data.")
    if keyframe_mode not in KEYFRAME_MODES:
        raise HTTPException(status_code=400, detail=f"keyframe_mode must be one of {list(KEYFRAME_MODES)}.")

    if not PIPELINE_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another video is already being processed. Please wait for it to finish.")
    try:
        return _process_uploaded_video(upload_file, sharpness_threshold, similarity_threshold, keyframe_mode)
    finally:
        PIPELINE_LOCK.release()


def _process_uploaded_video(upload_file: UploadFile, sharpness_threshold: float, similarity_threshold: float,
                            keyframe_mode: str):
    filename = upload_file.filename
    clean_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
    target_path = os.path.join(VIDEOS_DIR, clean_filename)

    # Save uploaded file
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    print(f"[API] Uploaded video saved: {target_path} ({os.path.getsize(target_path)} bytes)")

    # Execute full pipeline
    try:
        results = process_video_pipeline(
            video_path=target_path,
            filename=clean_filename,
            storage_dirs=STORAGE_CONFIG,
            sharpness_threshold=sharpness_threshold,
            similarity_threshold=similarity_threshold,
            keyframe_mode=keyframe_mode
        )
        save_cached_results(results)

        # Record the run to the document store
        try:
            get_store().insert("pipeline_jobs", {
                "job_id": f"JOB-{int(datetime.utcnow().timestamp())}",
                "filename": clean_filename,
                "total_frames": results["total_frames"],
                "sharp_frames": results["number_of_sharp_frames"],
                "keyframes": results["number_of_final_keyframes"],
                "trajectory_points": results["trajectory_points"],
                "resolution": results["resolution"],
                "fps": results["fps"]
            })
        except Exception as e:
            print(f"[DB LOG ERROR] {e}")

        return JSONResponse(status_code=200, content=results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        # A failed run must not leave an unprocessable file behind as a phantom "READY" project
        try:
            os.remove(target_path)
        except OSError:
            pass
        for base in (FRAMES_DIR, KEYFRAMES_DIR):
            shutil.rmtree(os.path.join(base, project_slug(clean_filename)), ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {e}")

@app.get("/api/progress")
async def get_pipeline_progress():
    """
    Returns live progress of video processing pipeline.
    """
    return get_current_progress()

# Backward-compatible upload endpoint
@app.post("/api/video/upload")
def legacy_upload_video(
    video: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    return process_video_endpoint(video=video, file=file)

# ==========================================
# 2. FRAME & KEYFRAME ACCESS ENDPOINTS
# ==========================================

@app.get("/api/frames/{frame_index}")
async def get_frame_image_by_index(frame_index: int):
    """
    Returns the exact image for extracted frame at frame_index (0-indexed).
    """
    filename = f"frame_{frame_index:04d}.jpg"
    filepath = os.path.join(active_frames_dir(), filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Frame {frame_index} not found.")

    return FileResponse(filepath, media_type="image/jpeg")

@app.get("/api/keyframes/{keyframe_index}")
async def get_keyframe_image_by_index(keyframe_index: int):
    """
    Returns the exact image for keyframe at keyframe_index (0-indexed).
    Matches show_selected_keyframe:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    """
    if latest_results and "keyframes" in latest_results:
        kfs = latest_results["keyframes"]
        if 0 <= keyframe_index < len(kfs):
            filepath = kfs[keyframe_index].get("path")
            if filepath and os.path.exists(filepath):
                return FileResponse(filepath, media_type="image/jpeg")

            orig_frame = kfs[keyframe_index].get("frame_index")
            if orig_frame is not None:
                source_path = os.path.join(active_frames_dir(), f"frame_{orig_frame:04d}.jpg")
                if os.path.exists(source_path):
                    return FileResponse(source_path, media_type="image/jpeg")

    # Check keyframes directory by prefix
    prefix = f"keyframe_{keyframe_index:04d}_"
    kf_dir = active_keyframes_dir()
    if os.path.exists(kf_dir):
        for fname in sorted(os.listdir(kf_dir)):
            if fname.startswith(prefix):
                return FileResponse(os.path.join(kf_dir, fname), media_type="image/jpeg")

    # Fallback to selected_frames
    if latest_results and "selected_frames" in latest_results and keyframe_index < len(latest_results["selected_frames"]):
        orig_frame = latest_results["selected_frames"][keyframe_index].get("frame_index", keyframe_index)
        frame_path = os.path.join(active_frames_dir(), f"frame_{orig_frame:04d}.jpg")
        if os.path.exists(frame_path):
            return FileResponse(frame_path, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Keyframe {keyframe_index} not found.")

@app.get("/api/video/selected_keyframe")
async def get_selected_keyframe_endpoint(
    index: int = Query(0, alias="keyframe_index"),
    mode: str = Query("visual")
):
    """
    Returns the exact metadata matching show_selected_keyframe(keyframe_index).
    Matches Colab notebook:
    frame_number, sharpness = selected_frames[keyframe_index]
    timestamp = frame_number / fps
    title = KEYFRAME {keyframe_index + 1} / {len(selected_frames)} ...
    """
    if not latest_results:
        raise HTTPException(status_code=404, detail="No video pipeline results loaded yet.")

    kfs_list = latest_results.get("selected_frames") if mode == "selected" and "selected_frames" in latest_results else latest_results.get("keyframes", [])
    if not kfs_list and "keyframes" in latest_results:
        kfs_list = latest_results["keyframes"]

    if not kfs_list:
        raise HTTPException(status_code=404, detail="No keyframes available.")

    idx = max(0, min(index, len(kfs_list) - 1))
    kf = kfs_list[idx]
    total_frames = latest_results.get("total_frames", len(kfs_list))
    fps = latest_results.get("fps", 30.0)

    frame_number = kf.get("frame_number", kf.get("frame_index", 0) + 1)
    sharpness = kf.get("sharpness", 0.0)
    timestamp = kf.get("timestamp", round((frame_number - 1) / fps, 2) if fps > 0 else 0.0)

    title = (
        f"KEYFRAME {idx + 1} / {len(kfs_list)}\n"
        f"Original Frame: {frame_number} / {total_frames}   |   "
        f"Time: {timestamp:.2f} sec   |   "
        f"Sharpness: {sharpness:.2f}"
    )

    return {
        "success": True,
        "keyframe_index": idx,
        "keyframe_number": idx + 1,
        "total_keyframes": len(kfs_list),
        "frame_number": frame_number,
        "original_frame_number": frame_number,
        "total_frames": total_frames,
        "timestamp": timestamp,
        "timestamp_sec": timestamp,
        "sharpness": sharpness,
        "title": title,
        "url": kf.get("url", f"/api/keyframes/{idx}"),
        "image_url": kf.get("url", f"/api/keyframes/{idx}")
    }

@app.get("/api/video/frame_image")
async def get_video_frame_image(frame: int = 0, width: int = 0):
    """
    Returns frame image by query param, downscaled to `width` px when it is larger
    (stored frames are full resolution). Falls back to the source video if the JPEG is gone.
    """
    if frame < 0:
        raise HTTPException(status_code=404, detail=f"Frame {frame} not found.")

    filepath = os.path.join(active_frames_dir(), f"frame_{frame:04d}.jpg")
    if width <= 0 and os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")

    img = load_frame(frame)
    if img is None:
        raise HTTPException(status_code=404, detail=f"Frame {frame} not found.")

    if width > 0 and img.shape[1] > width:
        new_h = max(1, int(img.shape[0] * (width / img.shape[1])))
        img = cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)
    elif os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")

@app.get("/api/video/frame_info")
async def get_video_frame_info(frame: int = 0):
    """
    Returns detailed frame info: frame number, timestamp, and sharpness score.
    """
    if latest_results and "frames" in latest_results:
        frames_list = latest_results["frames"]
        if 0 <= frame < len(frames_list):
            return frames_list[frame]

    sharpness = 0.0
    img = load_frame(frame)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    fps = latest_results.get("fps", 30.0) if latest_results else 30.0
    t_sec = round(frame / fps, 2) if fps > 0 else 0.0

    return {
        "frame_index": frame,
        "frame_number": frame + 1,
        "timestamp": t_sec,
        "sharpness": round(sharpness, 2),
        "url": f"/api/frames/{frame}"
    }

# ==========================================
# 3. METADATA, KEYFRAMES & TRAJECTORY APIs
# ==========================================

@app.get("/api/video/metadata")
async def get_video_metadata_endpoint():
    """
    Returns the metadata of the currently processed video.
    """
    if latest_results:
        return {
            "success": True,
            "filename": latest_results.get("filename"),
            "resolution": latest_results.get("resolution"),
            "width": latest_results.get("width"),
            "height": latest_results.get("height"),
            "fps": latest_results.get("fps"),
            "duration": latest_results.get("duration"),
            "duration_seconds": latest_results.get("duration"),
            "total_frames": latest_results.get("total_frames"),
            "number_of_extracted_frames": latest_results.get("number_of_extracted_frames"),
            "number_of_sharp_frames": latest_results.get("number_of_sharp_frames"),
            "number_of_final_keyframes": latest_results.get("number_of_final_keyframes"),
            "trajectory_points": latest_results.get("trajectory_points"),
            "successful_trajectory_matches": latest_results.get("successful_trajectory_matches"),
            "failed_trajectory_frames": latest_results.get("failed_trajectory_frames"),
            "sharpness_summary": latest_results.get("sharpness_summary", {})
        }

    return {
        "success": False,
        "filename": "No video processed yet",
        "resolution": "0 x 0",
        "width": 0,
        "height": 0,
        "fps": 0,
        "duration": 0,
        "total_frames": 0,
        "number_of_extracted_frames": 0,
        "number_of_sharp_frames": 0,
        "number_of_final_keyframes": 0,
        "trajectory_points": 0
    }

@app.get("/api/video/keyframes")
async def get_video_keyframes_endpoint(
    count: Optional[int] = None,
    mode: str = Query("visual")
):
    """
    Returns all final keyframes (or top N if count specified).
    mode='selected': returns all sharp frames (Snippet 2)
    mode='visual': returns visual redundancy keyframes (Snippet 3 & 4)
    """
    if latest_results:
        kfs = latest_results.get("selected_frames") if mode == "selected" and "selected_frames" in latest_results else latest_results.get("keyframes", [])
        if not kfs and "keyframes" in latest_results:
            kfs = latest_results["keyframes"]

        total_kfs = len(kfs)
        if count and count < len(kfs):
            kfs = kfs[:count]

        return {
            "success": True,
            "count": len(kfs),
            "total": total_kfs,
            "total_keyframes": total_kfs,
            "total_frames": latest_results.get("total_frames", total_kfs),
            "keyframes": kfs,
            "sharpness_summary": latest_results.get("sharpness_summary", {})
        }

    # If no cached results, look in keyframes dir
    kf_dir = active_keyframes_dir()
    files = sorted([f for f in os.listdir(kf_dir) if f.endswith(".jpg")]) if os.path.exists(kf_dir) else []
    kfs = []
    for idx, fname in enumerate(files):
        kfs.append({
            "keyframe_index": idx,
            "keyframe_number": idx + 1,
            "frame_number": idx + 1,
            "original_frame_number": idx + 1,
            "timestamp": round(idx * 0.5, 2),
            "sharpness": 120.0,
            "filename": fname,
            "url": f"/api/keyframes/{idx}",
            "image_url": f"/api/keyframes/{idx}"
        })
    return {
        "success": True,
        "count": len(kfs),
        "total": len(kfs),
        "total_keyframes": len(kfs),
        "keyframes": kfs
    }

@app.get("/api/trajectory")
async def get_trajectory_endpoint():
    """
    Returns camera trajectory calculated on ALL extracted frames.
    """
    if latest_results and "trajectory" in latest_results:
        return {
            "success": True,
            "total_frames": latest_results.get("total_frames"),
            "trajectory_points": latest_results.get("trajectory_points"),
            "successful_matches": latest_results.get("successful_trajectory_matches"),
            "failed_frames": latest_results.get("failed_trajectory_frames"),
            "trajectory": latest_results.get("trajectory", [])
        }

    slug = _active_slug()
    traj_path = os.path.join(TRAJECTORIES_DIR, f"{slug}.json") if slug else os.path.join(TRAJECTORIES_DIR, "trajectory.json")
    if os.path.exists(traj_path):
        with open(traj_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"success": True, **data}

    return {
        "success": False,
        "message": "No trajectory computed yet.",
        "trajectory": []
    }

# ==========================================
# 4. TELEMETRY & PIPELINE DATABASE (document store)
# ==========================================

@app.get("/api/health")
async def health_check():
    store = get_store()
    db_ok = store.healthy()
    counts = {"telemetry_logs": 0, "pipeline_jobs": 0}
    if db_ok:
        try:
            counts["telemetry_logs"] = store.count("telemetry_logs")
            counts["pipeline_jobs"] = store.count("pipeline_jobs")
        except Exception:
            db_ok = False

    total_extracted = len([f for f in os.listdir(active_frames_dir()) if f.endswith(".jpg")])
    total_kf = len([f for f in os.listdir(active_keyframes_dir()) if f.endswith(".jpg")])

    return {
        "status": "ONLINE",
        "backend": "FastAPI",
        "keyframe_modes": list(KEYFRAME_MODES),
        "default_keyframe_mode": DEFAULT_KEYFRAME_MODE,
        "db_backend": store.backend,
        "db_connected": db_ok,
        "database": DB_NAME,
        "collections": counts,
        "storage": {
            "extracted_frames_stored": total_extracted,
            "keyframes_stored": total_kf
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/api/telemetry")
async def log_telemetry(request: Request):
    try:
        data = await request.json()
        doc_id = get_store().insert("telemetry_logs", data)
        return {"success": True, "id": doc_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Failed to save telemetry."})

@app.get("/api/telemetry")
async def get_telemetry():
    try:
        logs = get_store().list("telemetry_logs", limit=20)
        return {"success": True, "count": len(logs), "logs": logs}
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Failed to read telemetry."})

@app.post("/api/pipeline/job")
async def save_pipeline_job(request: Request):
    try:
        data = await request.json()
        doc_id = get_store().insert("pipeline_jobs", data)
        return {"success": True, "id": doc_id}
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Failed to save job."})

@app.get("/api/pipeline/jobs")
async def get_pipeline_jobs():
    try:
        jobs = get_store().list("pipeline_jobs", limit=10)
        return {"success": True, "count": len(jobs), "jobs": jobs}
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Failed to read jobs."})

# ==========================================
# 4.5. PROJECTS DIRECTORY & SELECTION API
# ==========================================

@app.get("/api/projects")
async def get_projects_list():
    """
    Returns the list of all video projects (uploaded/processed)
    with their keyframes, trajectory, total_frames, duration, fps, resolution.
    """
    projects_dict = {}

    # 1. Inspect recorded pipeline jobs from the document store
    try:
        jobs = get_store().list("pipeline_jobs", limit=1000)
        for job in jobs:
            fname = job.get("filename")
            if fname and fname not in projects_dict:
                fps_val = float(job.get("fps") or 30.0)
                total_f = int(job.get("total_frames") or 0)
                dur_val = round(total_f / max(1.0, fps_val), 2)
                projects_dict[fname] = {
                    "filename": fname,
                    "resolution": job.get("resolution", "3840 x 2160"),
                    "fps": round(fps_val, 2),
                    "duration": dur_val,
                    "total_frames": total_f,
                    "keyframes": int(job.get("keyframes") or 0),
                    "trajectory_points": int(job.get("trajectory_points") or total_f),
                    "created_at": str(job.get("created_at", "")),
                    "status": "PROCESSED"
                }
    except Exception as e:
        print(f"[PROJECTS DB ERROR] {e}")

    # 2. Check storage/videos and uploads folders
    search_dirs = [VIDEOS_DIR, os.path.join(PROJECT_ROOT, "uploads")]
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for fname in os.listdir(sdir):
                if fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                    fpath = os.path.join(sdir, fname)
                    if fname not in projects_dict:
                        w, h, fps, total_f = 0, 0, 30.0, 0
                        dur = 0.0
                        try:
                            cap = cv2.VideoCapture(fpath)
                            if cap.isOpened():
                                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 30.0), 2)
                                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                dur = round(total_f / fps, 2) if fps > 0 else 0.0
                                cap.release()
                        except Exception:
                            pass
                        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).isoformat()
                        projects_dict[fname] = {
                            "filename": fname,
                            "resolution": f"{w} x {h}" if w and h else "3840 x 2160",
                            "fps": fps,
                            "duration": dur,
                            "total_frames": total_f,
                            "keyframes": total_f,
                            "trajectory_points": total_f,
                            "created_at": mtime,
                            "status": "READY"
                        }

    # 3. Check individual per-project results_*.json files
    if os.path.exists(STORAGE_DIR):
        for f in os.listdir(STORAGE_DIR):
            if f.startswith("results_") and f.endswith(".json"):
                try:
                    with open(os.path.join(STORAGE_DIR, f), "r", encoding="utf-8") as rf:
                        rdata = json.load(rf)
                        rfname = rdata.get("filename")
                        if rfname:
                            if rfname not in projects_dict:
                                projects_dict[rfname] = {}
                            projects_dict[rfname].update({
                                "filename": rfname,
                                "resolution": rdata.get("resolution", "3840 x 2160"),
                                "fps": rdata.get("fps", 30.0),
                                "duration": rdata.get("duration", 0.0),
                                "total_frames": rdata.get("total_frames", 0),
                                "keyframes": rdata.get("number_of_final_keyframes", len(rdata.get("keyframes", []))),
                                "trajectory_points": rdata.get("trajectory_points", len(rdata.get("trajectory", []))),
                                "status": "PROCESSED"
                            })
                except Exception:
                    pass

    # 4. Reflect current active results
    if latest_results and "filename" in latest_results:
        act_fname = latest_results.get("filename")
        if act_fname in projects_dict:
            projects_dict[act_fname].update({
                "resolution": latest_results.get("resolution", projects_dict[act_fname]["resolution"]),
                "fps": latest_results.get("fps", projects_dict[act_fname]["fps"]),
                "duration": latest_results.get("duration", projects_dict[act_fname]["duration"]),
                "total_frames": latest_results.get("total_frames", projects_dict[act_fname]["total_frames"]),
                "keyframes": latest_results.get("number_of_final_keyframes", projects_dict[act_fname]["keyframes"]),
                "trajectory_points": latest_results.get("trajectory_points", projects_dict[act_fname]["trajectory_points"]),
                "status": "PROCESSED"
            })

    current_active_file = latest_results.get("filename") if latest_results else None
    result_list = []
    for fname, p in projects_dict.items():
        safe_name = "".join(c for c in fname if c.isalnum() or c in "._- ")
        mtime_ts = 0.0
        candidate_paths = [
            os.path.join(STORAGE_DIR, f"results_{safe_name}.json"),
            os.path.join(VIDEOS_DIR, safe_name),
            os.path.join(PROJECT_ROOT, "uploads", safe_name),
            os.path.join(STORAGE_DIR, "latest_pipeline_results.json") if latest_results and latest_results.get("filename") == fname else None
        ]
        for cpath in candidate_paths:
            if cpath and os.path.exists(cpath):
                file_mtime = os.path.getmtime(cpath)
                if file_mtime > mtime_ts:
                    mtime_ts = file_mtime

        if mtime_ts == 0.0:
            if p.get("created_at"):
                try:
                    dt = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
                    mtime_ts = dt.timestamp()
                except Exception:
                    mtime_ts = datetime.utcnow().timestamp()
            else:
                mtime_ts = datetime.utcnow().timestamp()

        dt = datetime.fromtimestamp(mtime_ts)
        p["last_modified_timestamp"] = mtime_ts
        p["last_modified_iso"] = dt.isoformat()
        p["last_modified"] = dt.strftime("%d %b %Y, %I:%M:%S %p")
        p["is_active"] = (fname == current_active_file)
        result_list.append(p)

    # Sort in chronological order (most recently modified first)
    result_list.sort(key=lambda x: x.get("last_modified_timestamp", 0), reverse=True)

    return {"success": True, "count": len(result_list), "projects": result_list}


@app.post("/api/projects/select")
@app.get("/api/projects/select")
async def select_project_endpoint(filename: str = Query(...)):
    """
    Selects a video project by filename, loads its results into active state,
    and returns its keyframes, trajectory, total_frames, duration, fps, resolution.
    """
    global latest_results
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    proj_cache = get_project_cache_path(filename)
    
    loaded_data = None
    if os.path.exists(proj_cache):
        try:
            with open(proj_cache, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
        except Exception as e:
            print(f"[SELECT PROJECT CACHE READ ERROR] {e}")

    if not loaded_data and latest_results and latest_results.get("filename") == filename:
        loaded_data = latest_results

    # Fallback to checking video files directly
    if not loaded_data:
        target_video = os.path.join(VIDEOS_DIR, safe_name)
        if not os.path.exists(target_video):
            alt_target = os.path.join(PROJECT_ROOT, "uploads", safe_name)
            if os.path.exists(alt_target):
                target_video = alt_target

        if os.path.exists(target_video):
            cap = cv2.VideoCapture(target_video)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 30.0), 2)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = round(total / fps, 2) if fps > 0 else 0.0
            cap.release()
            loaded_data = {
                "success": True,
                "filename": filename,
                "resolution": f"{w} x {h}",
                "width": w,
                "height": h,
                "fps": fps,
                "duration": dur,
                "duration_seconds": dur,
                "total_frames": total,
                "number_of_extracted_frames": total,
                "number_of_sharp_frames": total,
                "number_of_final_keyframes": total,
                "trajectory_points": total,
                "successful_trajectory_matches": total,
                "failed_trajectory_frames": 0,
                "keyframes": [],
                "trajectory": []
            }
        else:
            raise HTTPException(status_code=404, detail=f"Project video '{filename}' not found.")

    save_cached_results(loaded_data)

    return {
        "success": True,
        "message": f"Project '{filename}' selected successfully",
        "data": {
            "filename": loaded_data.get("filename"),
            "resolution": loaded_data.get("resolution"),
            "width": loaded_data.get("width"),
            "height": loaded_data.get("height"),
            "fps": loaded_data.get("fps"),
            "duration": loaded_data.get("duration"),
            "total_frames": loaded_data.get("total_frames"),
            "keyframes": loaded_data.get("number_of_final_keyframes", len(loaded_data.get("keyframes", []))),
            "trajectory_points": loaded_data.get("trajectory_points", len(loaded_data.get("trajectory", []))),
            "keyframes_list": loaded_data.get("keyframes", [])[:50],
            "trajectory": loaded_data.get("trajectory", [])
        }
    }


# ==========================================
# 4.8. STAGE 04: CAMERA POSE (COLMAP) + DEPTH  (heavy compute runs in Colab)
# ==========================================

def _resolve_project(project: Optional[str]):
    """(results, slug) for the named project, or the active one when `project` is omitted."""
    results = None
    if project:
        if latest_results and latest_results.get("filename") == project:
            results = latest_results
        else:
            cache = get_project_cache_path(project)
            if os.path.exists(cache):
                with open(cache, "r", encoding="utf-8") as f:
                    results = json.load(f)
    else:
        results = latest_results
    if not results or not results.get("keyframes"):
        raise HTTPException(status_code=404, detail="No processed project with keyframes. Upload and process a video first.")
    return results, results.get("project_slug") or project_slug(results["filename"])


def _stage04_dir(slug: str) -> str:
    return os.path.join(STAGE04_STORAGE, slug)


def _project_keyframes_dir(slug: str) -> str:
    per_project = os.path.join(KEYFRAMES_DIR, slug)
    return per_project if os.path.isdir(per_project) else KEYFRAMES_DIR


def _project_frames_dir(slug: str) -> str:
    per_project = os.path.join(FRAMES_DIR, slug)
    return per_project if os.path.isdir(per_project) else FRAMES_DIR


@app.get("/api/stage04/status")
def stage04_status(project: Optional[str] = None):
    results, slug = _resolve_project(project)
    summary_path = os.path.join(_stage04_dir(slug), "summary.json")
    summary = None
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    return {"success": True, "project": results["filename"], "project_slug": slug,
            "keyframes": len(results["keyframes"]), "results_imported": summary is not None,
            "summary": summary}


@app.get("/api/stage04/notebook")
def stage04_notebook():
    if not os.path.isfile(stage04_service.NOTEBOOK_PATH):
        raise HTTPException(status_code=404, detail="Colab notebook is missing from the backend.")
    return FileResponse(stage04_service.NOTEBOOK_PATH, media_type="application/x-ipynb+json",
                        filename="aero3d_stage04_colab.ipynb")


@app.get("/api/stage04/export")
def stage04_export(project: Optional[str] = None):
    """Builds and downloads the package (keyframes + manifest + scripts) to run in Colab."""
    results, slug = _resolve_project(project)
    out_zip = os.path.join(_stage04_dir(slug), f"aero3d_stage04_{slug}.zip")
    try:
        stage04_service.export_package(results, _project_keyframes_dir(slug), _project_frames_dir(slug), out_zip, slug)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return FileResponse(out_zip, media_type="application/zip", filename=os.path.basename(out_zip))


@app.get("/api/stage04/export_info")
def stage04_export_info(project: Optional[str] = None):
    """Same checks as the export, without building the zip: keyframe count, gap warnings."""
    results, slug = _resolve_project(project)
    kfs = results["keyframes"]
    on_disk = [k for k in kfs if k.get("filename") and os.path.isfile(os.path.join(_project_keyframes_dir(slug), k["filename"]))]
    gaps = stage04_service._gap_stats(on_disk) if on_disk else {}
    fills = stage04_service.plan_gap_fills(on_disk, results.get("frames") or [], _project_frames_dir(slug)) if on_disk else []
    return {"success": True, "project": results["filename"], "keyframes": len(kfs), "keyframes_on_disk": len(on_disk),
            "gap_fill_frames": len(fills),
            "ready": len(on_disk) >= stage04_service.MIN_KEYFRAMES, "keyframe_gaps": gaps,
            "min_keyframes": stage04_service.MIN_KEYFRAMES}


@app.post("/api/stage04/import")
def stage04_import(file: UploadFile = File(...), project: Optional[str] = None):
    """Imports stage04_results.zip produced by the Colab notebook."""
    results, slug = _resolve_project(project)
    stage_dir = _stage04_dir(slug)
    os.makedirs(stage_dir, exist_ok=True)
    incoming = os.path.join(stage_dir, "incoming.zip")
    try:
        with open(incoming, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        summary = stage04_service.import_results(stage_dir, slug, len(results["keyframes"]), incoming)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.remove(incoming)
        except OSError:
            pass
    return {"success": True, "summary": summary}


@app.get("/api/stage04/scene")
def stage04_scene(project: Optional[str] = None, max_points: int = 25000):
    _, slug = _resolve_project(project)
    try:
        scene = stage04_service.load_scene(_stage04_dir(slug), max(1000, min(max_points, 100000)))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"success": True, **scene}


@app.get("/api/stage04/depth/{frame_index}")
def stage04_depth_preview(frame_index: int, project: Optional[str] = None):
    """Depth preview for a video frame index (0-based); covers keyframes and gap-fill frames."""
    _, slug = _resolve_project(project)
    path = stage04_service.depth_preview_path(_stage04_dir(slug), frame_index)
    if not path:
        raise HTTPException(status_code=404, detail=f"No depth map for frame {frame_index}.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/stage04/depth_info/{frame_index}")
def stage04_depth_info(frame_index: int, project: Optional[str] = None):
    _, slug = _resolve_project(project)
    info = stage04_service.depth_info(_stage04_dir(slug), frame_index)
    if not info:
        raise HTTPException(status_code=404, detail=f"No depth report entry for frame {frame_index}.")
    return {"success": True, **info}


# ==========================================
# 5. STATIC FILES & FRONTEND HOSTING
# ==========================================

app.mount("/assets", StaticFiles(directory=os.path.join(PROJECT_ROOT, "assets")), name="assets")
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
if os.path.exists(UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(PROJECT_ROOT, "index.html"))

ROOT_STATIC_EXTENSIONS = {".html", ".ico", ".png", ".svg", ".jpg", ".webmanifest"}

@app.get("/{filename}")
async def serve_root_file(filename: str):
    # Only plain static frontend files directly inside the project root; never source,
    # git metadata or anything reached through backslashes / ".." segments.
    target = os.path.realpath(os.path.join(PROJECT_ROOT, filename))
    if (os.path.dirname(target) == os.path.realpath(PROJECT_ROOT)
            and os.path.isfile(target)
            and os.path.splitext(target)[1].lower() in ROOT_STATIC_EXTENSIONS):
        return FileResponse(target)
    return FileResponse(os.path.join(PROJECT_ROOT, "index.html"))

if __name__ == "__main__":
    import uvicorn
    print("==================================================")
    print(" Aero3D Video Processing FastAPI Backend")
    print(f" Database: {DB_NAME} (backend: {get_store().backend})")
    print(" Serving at: http://localhost:8000")
    print("==================================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)
