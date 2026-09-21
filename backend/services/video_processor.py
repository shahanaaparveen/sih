import os
import re
import math
import json
import shutil
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from config import settings
from .keyframe_sfm import detect_scene_cuts, relative_sharpness_gate, select_sfm_keyframes

COMPARE_WIDTH = 320
COMPARE_HEIGHT = 180
SIMILARITY_THRESHOLD = 0.95
DEFAULT_SHARPNESS_THRESHOLD = 100.0
STORE_MAX_WIDTH = 1600   # stored frames are capped to this width (disk economy + COLMAP-friendly size)

# "notebook":  the original Colab filter (absolute sharpness cutoff + histogram correlation).
# "sfm":        overlap-based selection of the sharpest frame per window, accuracy first: consecutive
#               keyframes overlap 88-96%.
# "sfm_light":  the same with 65-85% overlap: about 4x fewer images, much faster COLMAP, a little less accurate.
# (see services/keyframe_sfm.py; presets chosen by measuring pose error against ground truth, see
#  scratch/compare_keyframe_modes.py)
SFM_PRESETS = {"sfm": (0.96, 0.88), "sfm_light": (0.85, 0.65)}   # (target_overlap, min_overlap)
KEYFRAME_MODES = ("notebook",) + tuple(SFM_PRESETS)
DEFAULT_KEYFRAME_MODE = "sfm"


def sfm_overlaps(mode: str, target_overlap: Optional[float] = None, min_overlap: Optional[float] = None) -> Tuple[float, float]:
    """Overlap band for an SfM mode; explicit values override the preset."""
    preset_target, preset_min = SFM_PRESETS[mode]
    return (preset_target if target_overlap is None else target_overlap,
            preset_min if min_overlap is None else min_overlap)

# Global thread-safe progress tracker for frontend status
pipeline_progress = {
    "status": "IDLE",
    "percent": 0,
    "current_frame": 0,
    "total_frames": 0,
    "stage": "Waiting for video upload",
    "message": ""
}

def get_current_progress() -> Dict[str, Any]:
    return dict(pipeline_progress)

def project_slug(filename: str) -> str:
    """
    Filesystem-safe per-project folder name, so every video keeps its own frames/keyframes.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename or "").strip(".")
    return safe or "video"

def get_video_metadata(video_path: str, filename: str) -> Dict[str, Any]:
    """
    Extracts video metadata using OpenCV cv2.VideoCapture.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(total_frames / fps) if fps > 0 else 0.0

    cap.release()

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "resolution": f"{width} x {height}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration": round(duration, 2),
        "duration_seconds": round(duration, 2)
    }

def prepare_frame_for_comparison(frame: np.ndarray) -> np.ndarray:
    """
    Resize to 320x180, convert to grayscale, and normalize with equalizeHist.
    Matches prepare_frame in user notebook Snippet 3.
    """
    resized = cv2.resize(frame, (COMPARE_WIDTH, COMPARE_HEIGHT))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def _run_pipeline(
    video_path: str,
    filename: str,
    storage_dirs: Dict[str, str],
    sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    keyframe_mode: str = DEFAULT_KEYFRAME_MODE,
    target_overlap: Optional[float] = None,
    min_overlap: Optional[float] = None
) -> Dict[str, Any]:
    """
    High-Performance Single-Pass Video Pipeline:
    Executes frame extraction, Laplacian sharpness scoring (exact Colab formula),
    keyframe selection (see KEYFRAME_MODES) and camera trajectory across ALL frames in ONE pass.
    """
    global pipeline_progress
    if keyframe_mode not in KEYFRAME_MODES:
        raise ValueError(f"keyframe_mode must be one of {KEYFRAME_MODES}, got {keyframe_mode!r}")
    slug = project_slug(filename)
    frames_dir = os.path.join(storage_dirs["frames"], slug)
    keyframes_dir = os.path.join(storage_dirs["keyframes"], slug)
    trajectories_dir = storage_dirs["trajectories"]

    # Re-processing a video replaces only that project's own frames and keyframes
    for d in [frames_dir, keyframes_dir]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    # Step 1: Video Metadata
    metadata = get_video_metadata(video_path, filename)
    fps = metadata["fps"] if metadata["fps"] > 0 else 30.0
    total_frames = max(1, metadata["total_frames"])
    orig_w = metadata["width"]
    orig_h = metadata["height"]

    pipeline_progress["status"] = "PROCESSING"
    pipeline_progress["percent"] = 5
    pipeline_progress["current_frame"] = 0
    pipeline_progress["total_frames"] = total_frames
    pipeline_progress["stage"] = "Initializing single-pass pipeline"
    pipeline_progress["message"] = f"Processing {total_frames} frames ({orig_w}x{orig_h} @ {fps}fps)"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        return _process_frames(cap, video_path, filename, slug, metadata, storage_dirs,
                               frames_dir, keyframes_dir, trajectories_dir,
                               sharpness_threshold, similarity_threshold,
                               keyframe_mode, target_overlap, min_overlap)
    finally:
        cap.release()


def _process_frames(cap, video_path, filename, slug, metadata, storage_dirs,
                    frames_dir, keyframes_dir, trajectories_dir,
                    sharpness_threshold, similarity_threshold,
                    keyframe_mode, target_overlap, min_overlap) -> Dict[str, Any]:
    global pipeline_progress
    fps = metadata["fps"] if metadata["fps"] > 0 else 30.0
    total_frames = max(1, metadata["total_frames"])
    orig_w = metadata["width"]
    orig_h = metadata["height"]

    # ORB Trajectory Detector setup
    work_w = 960 if orig_w > 960 else orig_w
    work_h = int(orig_h * (work_w / orig_w)) if orig_w > 0 else orig_h
    scale_factor = (orig_w / work_w) if work_w > 0 else 1.0

    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    trajectory: List[Dict[str, float]] = [{"x": 0.0, "y": 0.0}]
    successful_matches = 0
    failed_frames = 0

    frame_diffs = [0.0]         # frame_diffs[i]: mean abs difference between tiny greyscale frames i-1 and i (cut detection)
    prev_tiny = None
    frame_transforms = [None]   # frame_transforms[i]: 2x3 similarity mapping frame i-1 -> i (work resolution)
    frame_items = []
    frame_numbers = []
    blur_scores = []
    selected_sharp_frames = []
    visual_keyframes = []
    all_frame_paths = []

    import gc

    previous_keyframe_prep = None
    prev_kp = None
    prev_des = None

    # Frame budget: on long / 4K clips, sample every `stride`-th frame so we never write tens of
    # thousands of full-res JPEGs. Skipped frames are grabbed but NOT decoded (cheap). Sharpness and
    # trajectory are computed on the sampled frames; each keeps its ORIGINAL video index.
    budget = max(1, int(getattr(settings, "MAX_FRAMES_ON_DISK", 1500)))
    stride = max(1, math.ceil(total_frames / budget)) if total_frames > 0 else 1
    if stride > 1:
        print(f"[PIPELINE] {total_frames} frames exceed budget {budget}: sampling every {stride} frame(s)")

    pos = 0        # processed-frame position (aligned with blur_scores / frame_transforms / frame_diffs)
    raw = -1       # original video frame index

    while True:
        try:
            grabbed = cap.grab()
        except Exception as err:
            print(f"[PIPELINE] Video read ended or error near frame {raw}: {err}")
            break
        if not grabbed:
            break
        raw += 1
        if raw % stride != 0:
            continue                                  # frame budget: skip without decoding
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        frame_filename = f"frame_{raw:04d}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)

        # 1. Store a width-capped JPEG (disk economy + COLMAP-friendly). Sharpness below is still on full res.
        if frame.shape[1] > STORE_MAX_WIDTH:
            sh = max(1, int(frame.shape[0] * (STORE_MAX_WIDTH / frame.shape[1])))
            store_img = cv2.resize(frame, (STORE_MAX_WIDTH, sh), interpolation=cv2.INTER_AREA)
        else:
            store_img = frame
        cv2.imwrite(frame_path, store_img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        all_frame_paths.append(frame_path)

        # 2. Exact Laplacian Sharpness Score on Full-Resolution Grayscale (Matches Snippet 1)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())

        frame_numbers.append(raw)
        blur_scores.append(sharpness)

        t_sec = round(raw / fps, 2)
        frame_items.append({
            "frame_index": raw,
            "frame_number": raw + 1,
            "timestamp": t_sec,
            "sharpness": round(sharpness, 2),
            "filename": frame_filename,
            "url": f"/api/frames/{raw}"
        })

        # 3. Blur Filtering & Visual Redundancy Keyframe Selection (notebook mode; sfm selects after the pass)
        if keyframe_mode == "notebook" and sharpness >= sharpness_threshold:
            selected_sharp_frames.append((raw, sharpness))
            current_prep = prepare_frame_for_comparison(frame)
            if previous_keyframe_prep is None:
                visual_keyframes.append((raw, sharpness))
                previous_keyframe_prep = current_prep
            else:
                hist_prev = cv2.calcHist([previous_keyframe_prep], [0], None, [256], [0, 256])
                hist_curr = cv2.calcHist([current_prep], [0], None, [256], [0, 256])
                similarity = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
                if similarity < similarity_threshold:
                    visual_keyframes.append((raw, sharpness))
                    previous_keyframe_prep = current_prep

        # 4. Camera trajectory / motion between consecutive SAMPLED frames
        gray_work = cv2.resize(gray_full, (work_w, work_h)) if work_w != orig_w else gray_full
        curr_kp, curr_des = orb.detectAndCompute(gray_work, None)

        tiny = cv2.resize(gray_work, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev_tiny is not None:
            frame_diffs.append(float(np.abs(tiny - prev_tiny).mean()))
        prev_tiny = tiny

        frame_M = None
        if pos == 0:
            prev_kp = curr_kp
            prev_des = curr_des
        else:
            if prev_des is None or curr_des is None:
                trajectory.append(dict(trajectory[-1]))
                failed_frames += 1
            else:
                try:
                    matches = bf.match(prev_des, curr_des)
                    matches = sorted(matches, key=lambda m: m.distance)[:80]
                    if len(matches) >= 8:
                        pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                        pts2 = np.float32([curr_kp[m.trainIdx].pt for m in matches])
                        M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
                        if M is not None:
                            frame_M = M
                            dx = float(M[0, 2]) * scale_factor
                            dy = float(M[1, 2]) * scale_factor
                            new_x = float(trajectory[-1]["x"] + dx)
                            new_y = float(trajectory[-1]["y"] + dy)
                            trajectory.append({"x": round(new_x, 2), "y": round(new_y, 2)})
                            successful_matches += 1
                        else:
                            trajectory.append(dict(trajectory[-1]))
                            failed_frames += 1
                    else:
                        trajectory.append(dict(trajectory[-1]))
                        failed_frames += 1
                except Exception:
                    trajectory.append(dict(trajectory[-1]))
                    failed_frames += 1

            frame_transforms.append(frame_M)
            prev_kp = curr_kp
            prev_des = curr_des

        del frame, gray_full, gray_work
        pos += 1
        if pos % 100 == 0:
            gc.collect()

        # Periodic progress update
        if pos % 25 == 0:
            pct = min(95, int((raw / max(1, total_frames)) * 90) + 5)
            pipeline_progress["percent"] = pct
            pipeline_progress["current_frame"] = raw
            pipeline_progress["stage"] = f"Extracting, scoring sharpness & computing trajectory ({raw}/{total_frames})"
            pipeline_progress["message"] = f"{pct}% · frame {raw} of {total_frames} ({pos} kept)"

    cap.release()
    gc.collect()
    video_total = raw + 1 if raw >= 0 else 0
    frame_id = pos                    # number of frames actually processed/stored

    # Selection works on processed positions; map those back to original video frame indices.
    scene_cuts = [int(frame_numbers[p]) for p in detect_scene_cuts(frame_diffs) if 0 <= p < len(frame_numbers)]
    keyframe_overlap: Dict[int, Optional[float]] = {}
    keyframe_params: Dict[str, Any] = {}
    if keyframe_mode in SFM_PRESETS:
        target_overlap, min_overlap = sfm_overlaps(keyframe_mode, target_overlap, min_overlap)
        keyframe_params = {"target_overlap": target_overlap, "min_overlap": min_overlap}
        selection = select_sfm_keyframes(frame_transforms, blur_scores, work_w, work_h, target_overlap, min_overlap)
        visual_keyframes = [(frame_numbers[p], blur_scores[p]) for p in selection["keyframes"]]
        keyframe_overlap = {frame_numbers[p]: ov for p, ov in zip(selection["keyframes"], selection["overlaps"])}
        selected_sharp_frames = [(frame_numbers[p], blur_scores[p]) for p in relative_sharpness_gate(blur_scores)]
    # Adaptive fallback if no frames reached sharpness_threshold
    elif not selected_sharp_frames and blur_scores:
        median_score = float(np.median(blur_scores))
        selected_sharp_frames = [(fn, s) for fn, s in zip(frame_numbers, blur_scores) if s >= median_score]
        if not visual_keyframes:
            visual_keyframes = list(selected_sharp_frames)

    # 5. Build Final Visual Keyframe Records & Save to Storage
    keyframe_records = []
    for keyframe_index, (orig_frame_id, kf_sharpness) in enumerate(visual_keyframes):
        kf_filename = f"keyframe_{keyframe_index:04d}_original_{orig_frame_id:04d}.jpg"
        kf_path = os.path.join(keyframes_dir, kf_filename)
        source_frame_path = os.path.join(frames_dir, f"frame_{orig_frame_id:04d}.jpg")
        if os.path.exists(source_frame_path):
            shutil.copyfile(source_frame_path, kf_path)

        kf_time = round(orig_frame_id / fps, 2)
        title = (
            f"KEYFRAME {keyframe_index + 1} / {len(visual_keyframes)}\n"
            f"Original Frame: {orig_frame_id + 1} / {total_frames}   |   "
            f"Time: {kf_time:.2f} sec   |   "
            f"Sharpness: {kf_sharpness:.2f}"
        )

        keyframe_records.append({
            "keyframe_index": keyframe_index,
            "keyframe_number": keyframe_index + 1,
            "frame_number": orig_frame_id + 1,
            "original_frame_number": orig_frame_id + 1,
            "frame_index": orig_frame_id,
            "timestamp": kf_time,
            "timestamp_sec": kf_time,
            "sharpness": round(kf_sharpness, 2),
            "title": title,
            "overlap_with_previous": (None if keyframe_overlap.get(orig_frame_id) is None
                                      else round(keyframe_overlap[orig_frame_id], 3)),
            "filename": kf_filename,
            "path": kf_path,
            "url": f"/api/keyframes/{keyframe_index}",
            "image_url": f"/api/keyframes/{keyframe_index}"
        })

    # 6. Build Selected Sharp Frames Records (Matches Snippet 2 show_selected_keyframe)
    selected_frames_records = []
    for keyframe_index, (orig_frame_id, kf_sharpness) in enumerate(selected_sharp_frames):
        kf_time = round(orig_frame_id / fps, 2)
        title = (
            f"KEYFRAME {keyframe_index + 1} / {len(selected_sharp_frames)}\n"
            f"Original Frame: {orig_frame_id + 1} / {total_frames}   |   "
            f"Time: {kf_time:.2f} sec   |   "
            f"Sharpness: {kf_sharpness:.2f}"
        )
        selected_frames_records.append({
            "keyframe_index": keyframe_index,
            "keyframe_number": keyframe_index + 1,
            "frame_number": orig_frame_id + 1,
            "original_frame_number": orig_frame_id + 1,
            "frame_index": orig_frame_id,
            "timestamp": kf_time,
            "timestamp_sec": kf_time,
            "sharpness": round(kf_sharpness, 2),
            "title": title,
            "url": f"/api/frames/{orig_frame_id}",
            "image_url": f"/api/frames/{orig_frame_id}"
        })

    # Save trajectory to storage
    traj_json_path = os.path.join(trajectories_dir, f"{slug}.json")
    os.makedirs(trajectories_dir, exist_ok=True)
    with open(traj_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_frames": len(trajectory),
            "trajectory_points": len(trajectory),
            "successful_matches": successful_matches,
            "failed_frames": failed_frames,
            "points": trajectory
        }, f, indent=2)

    pipeline_progress["status"] = "COMPLETED"
    pipeline_progress["percent"] = 100
    pipeline_progress["stage"] = "Processing complete"
    pipeline_progress["message"] = f"100% · Processed all {frame_id} frames successfully"

    # Step 7: Final Pipeline Results
    sharpness_summary = {
        "frames_analyzed": len(frame_numbers),
        "min_score": round(min(blur_scores), 2) if blur_scores else 0.0,
        "max_score": round(max(blur_scores), 2) if blur_scores else 0.0,
        "avg_score": round(float(np.mean(blur_scores)), 2) if blur_scores else 0.0,
        "original_frames": len(frame_numbers),
        "after_blur_filtering": len(selected_sharp_frames),
        "before_visual_filtering": len(selected_sharp_frames),
        "after_visual_filtering": len(keyframe_records),
        "removed_redundant_frames": len(selected_sharp_frames) - len(keyframe_records)
    }

    results = {
        "success": True,
        "filename": filename,
        "project_slug": slug,
        "resolution": metadata["resolution"],
        "width": metadata["width"],
        "height": metadata["height"],
        "fps": metadata["fps"],
        "duration": metadata["duration"],
        "duration_seconds": metadata["duration"],
        "total_frames": video_total,                       # true video length
        "frame_sample_stride": stride,                     # 1 = every frame kept; >1 = frame budget applied
        "number_of_extracted_frames": len(all_frame_paths),
        "number_of_sharp_frames": len(selected_sharp_frames),
        "number_of_final_keyframes": len(keyframe_records),
        "trajectory_points": len(trajectory),
        "successful_trajectory_matches": successful_matches,
        "failed_trajectory_frames": failed_frames,
        "trajectory": trajectory,
        "keyframe_mode": keyframe_mode,
        "keyframe_params": keyframe_params,
        "scene_cuts": scene_cuts,
        "keyframes": keyframe_records,
        "selected_frames": selected_frames_records,
        "frames": frame_items,
        "sharpness_summary": sharpness_summary
    }

    print("================================")
    print("SHARPNESS ANALYSIS")
    print("================================")
    print("Frames analyzed :", sharpness_summary["frames_analyzed"])
    print("Minimum score   :", sharpness_summary["min_score"])
    print("Maximum score   :", sharpness_summary["max_score"])
    print("Average score   :", sharpness_summary["avg_score"])
    print("================================")
    print("KEYFRAME SELECTION")
    print("================================")
    print("Original frames          :", sharpness_summary["original_frames"])
    print("After blur filtering     :", sharpness_summary["after_blur_filtering"])
    print("Before visual filtering  :", sharpness_summary["before_visual_filtering"])
    print("After visual filtering   :", sharpness_summary["after_visual_filtering"])
    print("Removed redundant frames :", sharpness_summary["removed_redundant_frames"])

    return results


def process_video_pipeline(
    video_path: str,
    filename: str,
    storage_dirs: Dict[str, str],
    sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    keyframe_mode: str = DEFAULT_KEYFRAME_MODE,
    target_overlap: Optional[float] = None,
    min_overlap: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs the single-pass pipeline and keeps pipeline_progress truthful on failure.
    """
    try:
        return _run_pipeline(video_path, filename, storage_dirs, sharpness_threshold, similarity_threshold,
                             keyframe_mode, target_overlap, min_overlap)
    except Exception as err:
        pipeline_progress["status"] = "FAILED"
        pipeline_progress["percent"] = 0
        pipeline_progress["current_frame"] = 0
        pipeline_progress["stage"] = "Processing failed"
        pipeline_progress["message"] = str(err)
        raise
