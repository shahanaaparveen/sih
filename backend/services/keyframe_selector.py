import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# --------------------------------------------------
# SETTINGS (Matches Colab Notebook)
# --------------------------------------------------
COMPARE_WIDTH = 320
COMPARE_HEIGHT = 180
SIMILARITY_THRESHOLD = 0.95
DEFAULT_SHARPNESS_THRESHOLD = 100.0

# --------------------------------------------------
# 1. SHARPNESS ANALYSIS (Matches Snippet 1)
# --------------------------------------------------

def calculate_sharpness_from_frame(frame: np.ndarray) -> float:
    """
    Computes Laplacian variance sharpness score for a single frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharpness

def calculate_sharpness_from_video(video_path: str) -> Tuple[List[int], List[float]]:
    """
    Analyzes all video frames and returns frame numbers and blur scores.
    Exact match to Snippet 1 from user notebook.
    """
    cap = cv2.VideoCapture(video_path)
    frame_numbers: List[int] = []
    blur_scores: List[float] = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        frame_numbers.append(frame_id)
        blur_scores.append(sharpness)
        frame_id += 1

    cap.release()

    if blur_scores:
        print("✅ Sharpness analysis completed")
        print("--------------------------------")
        print("Frames analyzed :", len(frame_numbers))
        print("Minimum score   :", round(min(blur_scores), 2))
        print("Maximum score   :", round(max(blur_scores), 2))
        print("Average score   :", round(float(np.mean(blur_scores)), 2))

    return frame_numbers, blur_scores

def calculate_sharpness(frame_paths: List[str]) -> Tuple[List[int], List[float]]:
    """
    Calculates Laplacian sharpness variance for list of frame paths.
    """
    frame_numbers = []
    blur_scores = []

    for frame_number, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        sharpness = calculate_sharpness_from_frame(frame)
        frame_numbers.append(frame_number)
        blur_scores.append(sharpness)

    return frame_numbers, blur_scores

# --------------------------------------------------
# 2. BLUR FILTERING (SELECTED SHARP FRAMES)
# --------------------------------------------------

def filter_sharp_frames(
    frame_numbers: List[int],
    blur_scores: List[float],
    threshold: float = DEFAULT_SHARPNESS_THRESHOLD
) -> List[Tuple[int, float]]:
    """
    Filters frames meeting or exceeding the sharpness threshold.
    Returns: List of (frame_number, sharpness)
    """
    selected = [
        (fn, score)
        for fn, score in zip(frame_numbers, blur_scores)
        if score >= threshold
    ]

    # Adaptive fallback if no frames reach threshold
    if not selected and blur_scores:
        median_score = float(np.median(blur_scores))
        selected = [
            (fn, score)
            for fn, score in zip(frame_numbers, blur_scores)
            if score >= median_score
        ]
        if not selected:
            selected = list(zip(frame_numbers, blur_scores))

    return selected

# --------------------------------------------------
# 3. VISUAL REDUNDANCY FILTERING (Matches Snippet 3)
# --------------------------------------------------

def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """
    Resize to 320x180, convert to grayscale, and normalize with equalizeHist.
    Exact match to prepare_frame in Snippet 3.
    """
    resized = cv2.resize(frame, (COMPARE_WIDTH, COMPARE_HEIGHT))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def select_visual_keyframes_from_video(
    video_path: str,
    selected_frames: List[Tuple[int, float]],
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> List[Tuple[int, float]]:
    """
    Performs normalized correlation histogram comparison on selected sharp frames.
    Exact match to Snippet 3.
    """
    cap = cv2.VideoCapture(video_path)
    visual_keyframes: List[Tuple[int, float]] = []
    previous_frame = None

    for frame_number, sharpness in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        current_frame = prepare_frame(frame)

        # First frame is automatically selected
        if previous_frame is None:
            visual_keyframes.append((frame_number, sharpness))
            previous_frame = current_frame
            continue

        # Calculate structural similarity using normalized correlation
        similarity = cv2.compareHist(
            cv2.calcHist([previous_frame], [0], None, [256], [0, 256]),
            cv2.calcHist([current_frame], [0], None, [256], [0, 256]),
            cv2.HISTCMP_CORREL
        )

        # Keep only if sufficiently different
        if similarity < similarity_threshold:
            visual_keyframes.append((frame_number, sharpness))
            previous_frame = current_frame

    cap.release()
    return visual_keyframes

def select_keyframes(
    all_frame_paths: List[str],
    selected_sharp_frames: List[Tuple[int, float]],
    output_dir: str,
    fps: float = 30.0,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    total_video_frames: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Selects visual keyframes, saves them to output_dir, and builds metadata records.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_frames = total_video_frames or len(all_frame_paths)

    # Clear previous keyframes
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

    visual_keyframes: List[Tuple[int, float]] = []
    previous_frame = None

    for frame_number, sharpness in selected_sharp_frames:
        if frame_number >= len(all_frame_paths):
            continue

        frame_path = all_frame_paths[frame_number]
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        current_frame = prepare_frame(frame)

        if previous_frame is None:
            visual_keyframes.append((frame_number, sharpness))
            previous_frame = current_frame
            continue

        hist_previous = cv2.calcHist([previous_frame], [0], None, [256], [0, 256])
        hist_current = cv2.calcHist([current_frame], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(hist_previous, hist_current, cv2.HISTCMP_CORREL)

        if similarity < similarity_threshold:
            visual_keyframes.append((frame_number, sharpness))
            previous_frame = current_frame

    # Save keyframes
    keyframe_records: List[Dict[str, Any]] = []

    for keyframe_index, (frame_number, sharpness) in enumerate(visual_keyframes):
        source_path = all_frame_paths[frame_number]
        frame = cv2.imread(source_path)
        if frame is None:
            continue

        filename = f"keyframe_{keyframe_index:04d}_original_{frame_number:04d}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, frame)

        timestamp = float(frame_number / fps) if fps > 0 else 0.0

        title = (
            f"KEYFRAME {keyframe_index + 1} / {len(visual_keyframes)}\n"
            f"Original Frame: {frame_number + 1} / {total_frames}   |   "
            f"Time: {timestamp:.2f} sec   |   "
            f"Sharpness: {sharpness:.2f}"
        )

        keyframe_records.append({
            "keyframe_index": keyframe_index,
            "keyframe_number": keyframe_index + 1,
            "frame_number": frame_number + 1,
            "original_frame_number": frame_number + 1,
            "frame_index": frame_number,
            "timestamp": round(timestamp, 2),
            "timestamp_sec": round(timestamp, 2),
            "sharpness": round(float(sharpness), 2),
            "title": title,
            "filename": filename,
            "path": output_path,
            "url": f"/api/keyframes/{keyframe_index}",
            "image_url": f"/api/keyframes/{keyframe_index}"
        })

    print(f"[KEYFRAME_SELECTOR] Selected and saved {len(keyframe_records)} keyframes to {output_dir}")
    return keyframe_records

# --------------------------------------------------
# 4. SELECTED KEYFRAME VIEWER HELPER (Matches Snippet 2 & 4)
# --------------------------------------------------

def get_selected_keyframe_data(
    keyframe_index: int,
    selected_frames: List[Tuple[int, float]],
    total_frames: int,
    fps: float = 30.0,
    video_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns exact metadata corresponding to show_selected_keyframe(keyframe_index).
    """
    if not selected_frames:
        raise ValueError("selected_frames list is empty.")

    keyframe_index = max(0, min(keyframe_index, len(selected_frames) - 1))
    frame_number, sharpness = selected_frames[keyframe_index]
    timestamp = float(frame_number / fps) if fps > 0 else 0.0

    title = (
        f"KEYFRAME {keyframe_index + 1} / {len(selected_frames)}\n"
        f"Original Frame: {frame_number + 1} / {total_frames}   |   "
        f"Time: {timestamp:.2f} sec   |   "
        f"Sharpness: {sharpness:.2f}"
    )

    return {
        "keyframe_index": keyframe_index,
        "keyframe_number": keyframe_index + 1,
        "total_keyframes": len(selected_frames),
        "frame_number": frame_number + 1,
        "original_frame_number": frame_number + 1,
        "frame_index": frame_number,
        "total_frames": total_frames,
        "timestamp": round(timestamp, 2),
        "timestamp_sec": round(timestamp, 2),
        "sharpness": round(float(sharpness), 2),
        "title": title,
        "url": f"/api/keyframes/{keyframe_index}",
        "image_url": f"/api/keyframes/{keyframe_index}"
    }

def read_frame_at_index(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Reads original frame from video using cv2.VideoCapture and CAP_PROP_POS_FRAMES.
    Matches Snippet 2:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    """
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame
