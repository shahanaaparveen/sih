import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

COMPARE_WIDTH = 320
COMPARE_HEIGHT = 180
SIMILARITY_THRESHOLD = 0.95
DEFAULT_SHARPNESS_THRESHOLD = 100.0

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

def process_video_pipeline(
    video_path: str,
    filename: str,
    storage_dirs: Dict[str, str],
    sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """
    High-Performance Single-Pass Video Pipeline:
    Executes frame extraction, Laplacian sharpness scoring (exact Colab formula),
    visual redundancy keyframe filtering (histogram correlation < SIMILARITY_THRESHOLD),
    and camera trajectory across ALL frames in ONE single in-memory pass.
    """
    global pipeline_progress
    frames_dir = storage_dirs["frames"]
    keyframes_dir = storage_dirs["keyframes"]
    trajectories_dir = storage_dirs["trajectories"]

    # Clear previous frames and keyframes
    for d in [frames_dir, keyframes_dir]:
        os.makedirs(d, exist_ok=True)
        for fname in os.listdir(d):
            fpath = os.path.join(d, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass

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

    # ORB Trajectory Detector setup
    work_w = 960 if orig_w > 960 else orig_w
    work_h = int(orig_h * (work_w / orig_w)) if orig_w > 0 else orig_h
    scale_factor = (orig_w / work_w) if work_w > 0 else 1.0

    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    trajectory: List[Dict[str, float]] = [{"x": 0.0, "y": 0.0}]
    successful_matches = 0
    failed_frames = 0

    frame_items = []
    frame_numbers = []
    blur_scores = []
    selected_sharp_frames = []
    visual_keyframes = []
    all_frame_paths = []

    import gc
    import shutil

    previous_keyframe_prep = None
    prev_kp = None
    prev_des = None

    frame_id = 0

    while True:
        try:
            ret, frame = cap.read()
        except Exception as err:
            print(f"[PIPELINE] Video read ended or error at frame {frame_id}: {err}")
            break

        if not ret or frame is None:
            break

        frame_filename = f"frame_{frame_id:04d}.jpg"
        frame_path = os.path.join(frames_dir, frame_filename)

        # 1. Save frame to disk for slider inspection
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        all_frame_paths.append(frame_path)

        # 2. Exact Laplacian Sharpness Score on Full-Resolution Grayscale (Matches Snippet 1)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())

        frame_numbers.append(frame_id)
        blur_scores.append(sharpness)

        t_sec = round(frame_id / fps, 2)
        frame_items.append({
            "frame_index": frame_id,
            "frame_number": frame_id + 1,
            "timestamp": t_sec,
            "sharpness": round(sharpness, 2),
            "filename": frame_filename,
            "url": f"/api/frames/{frame_id}"
        })

        # 3. Blur Filtering & Visual Redundancy Keyframe Selection (Matches Snippet 2 & 3)
        if sharpness >= sharpness_threshold:
            selected_sharp_frames.append((frame_id, sharpness))

            # Visual redundancy filtering using normalized histogram correlation
            current_prep = prepare_frame_for_comparison(frame)

            if previous_keyframe_prep is None:
                # First frame is automatically selected
                visual_keyframes.append((frame_id, sharpness))
                previous_keyframe_prep = current_prep
            else:
                hist_prev = cv2.calcHist([previous_keyframe_prep], [0], None, [256], [0, 256])
                hist_curr = cv2.calcHist([current_prep], [0], None, [256], [0, 256])
                similarity = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

                if similarity < similarity_threshold:
                    visual_keyframes.append((frame_id, sharpness))
                    previous_keyframe_prep = current_prep

        # 4. Camera Trajectory Estimation on ALL frames
        gray_work = cv2.resize(gray_full, (work_w, work_h)) if work_w != orig_w else gray_full
        curr_kp, curr_des = orb.detectAndCompute(gray_work, None)

        if frame_id == 0:
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

            prev_kp = curr_kp
            prev_des = curr_des

        # Cleanup frame buffer
        del frame
        del gray_full
        del gray_work

        frame_id += 1

        if frame_id % 100 == 0:
            gc.collect()

        # Periodic progress update
        if frame_id % 25 == 0 or frame_id == total_frames:
            pct = min(95, int((frame_id / total_frames) * 90) + 5)
            pipeline_progress["percent"] = pct
            pipeline_progress["current_frame"] = frame_id
            pipeline_progress["stage"] = f"Extracting, scoring sharpness & computing trajectory ({frame_id}/{total_frames})"
            pipeline_progress["message"] = f"{pct}% · {frame_id} of {total_frames} frames processed"

    cap.release()
    gc.collect()

    # Adaptive fallback if no frames reached sharpness_threshold
    if not selected_sharp_frames and blur_scores:
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
    traj_json_path = os.path.join(trajectories_dir, "trajectory.json")
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
        "resolution": metadata["resolution"],
        "width": metadata["width"],
        "height": metadata["height"],
        "fps": metadata["fps"],
        "duration": metadata["duration"],
        "duration_seconds": metadata["duration"],
        "total_frames": frame_id,
        "number_of_extracted_frames": len(all_frame_paths),
        "number_of_sharp_frames": len(selected_sharp_frames),
        "number_of_final_keyframes": len(keyframe_records),
        "trajectory_points": len(trajectory),
        "successful_trajectory_matches": successful_matches,
        "failed_trajectory_frames": failed_frames,
        "trajectory": trajectory,
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
