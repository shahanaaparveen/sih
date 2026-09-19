import os
import json
import cv2
import numpy as np
import shutil
import time

FRAMES_DIR = "backend/storage/frames"
KEYFRAMES_DIR = "backend/storage/keyframes"
CACHE_FILE = "backend/storage/latest_pipeline_results.json"

total_frames = 712
fps = 29.97

print(f"[START] Processing {total_frames} frames for exact sharpness & keyframe selection...")
t0 = time.time()

frame_numbers = []
blur_scores = []
frame_items = []

for frame_id in range(total_frames):
    frame_filename = f"frame_{frame_id:04d}.jpg"
    frame_path = os.path.join(FRAMES_DIR, frame_filename)
    
    # Fast direct grayscale load
    gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
        
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_scores.append(sharpness)
    frame_numbers.append(frame_id)
    
    t_sec = round(frame_id / fps, 2)
    frame_items.append({
        "frame_index": frame_id,
        "frame_number": frame_id + 1,
        "timestamp": t_sec,
        "sharpness": round(sharpness, 2),
        "filename": frame_filename,
        "url": f"/api/frames/{frame_id}"
    })
    
    del gray
    if (frame_id + 1) % 100 == 0:
        print(f"  Processed {frame_id + 1}/{total_frames} frames (elapsed: {time.time() - t0:.1f}s)...")

print("================================")
print("SHARPNESS ANALYSIS")
print("================================")
print("Frames analyzed :", len(frame_numbers))
print("Minimum score   :", round(min(blur_scores), 2))
print("Maximum score   :", round(max(blur_scores), 2))
print("Average score   :", round(float(np.mean(blur_scores)), 2))

# Blur Filtering
SHARPNESS_THRESHOLD = 100.0
selected_frames = [
    (fn, score)
    for fn, score in zip(frame_numbers, blur_scores)
    if score >= SHARPNESS_THRESHOLD
]

# Adaptive fallback if needed
if not selected_frames:
    median_score = float(np.median(blur_scores))
    selected_frames = [(fn, s) for fn, s in zip(frame_numbers, blur_scores) if s >= median_score]

print("================================")
print("BLUR FILTERING")
print("================================")
print("Original frames       :", len(frame_numbers))
print("Sharp frames selected :", len(selected_frames))

# Visual Redundancy Filtering (Snippet 3)
COMPARE_WIDTH = 320
COMPARE_HEIGHT = 180
SIMILARITY_THRESHOLD = 0.95

def prepare_gray_frame(gray):
    resized = cv2.resize(gray, (COMPARE_WIDTH, COMPARE_HEIGHT))
    equalized = cv2.equalizeHist(resized)
    return equalized

visual_keyframes = []
previous_prep = None

for frame_number, sharpness in selected_frames:
    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_number:04d}.jpg")
    gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        continue
        
    current_prep = prepare_gray_frame(gray)
    del gray
    
    if previous_prep is None:
        visual_keyframes.append((frame_number, sharpness))
        previous_prep = current_prep
        continue
        
    hist_previous = cv2.calcHist([previous_prep], [0], None, [256], [0, 256])
    hist_current = cv2.calcHist([current_prep], [0], None, [256], [0, 256])
    similarity = cv2.compareHist(hist_previous, hist_current, cv2.HISTCMP_CORREL)
    
    if similarity < SIMILARITY_THRESHOLD:
        visual_keyframes.append((frame_number, sharpness))
        previous_prep = current_prep

print("================================")
print("KEYFRAME SELECTION")
print("================================")
print("Original frames          :", len(frame_numbers))
print("After blur filtering     :", len(selected_frames))
print("Final keyframes          :", len(visual_keyframes))
print("Removed redundant frames :", len(selected_frames) - len(visual_keyframes))

# Save keyframes
os.makedirs(KEYFRAMES_DIR, exist_ok=True)
# Clear old keyframes
for f in os.listdir(KEYFRAMES_DIR):
    fp = os.path.join(KEYFRAMES_DIR, f)
    if os.path.isfile(fp):
        try: os.remove(fp)
        except: pass

keyframe_records = []
for keyframe_index, (frame_number, sharpness) in enumerate(visual_keyframes):
    kf_filename = f"keyframe_{keyframe_index:04d}_original_{frame_number:04d}.jpg"
    kf_path = os.path.join(KEYFRAMES_DIR, kf_filename)
    source_path = os.path.join(FRAMES_DIR, f"frame_{frame_number:04d}.jpg")
    
    if os.path.exists(source_path):
        shutil.copyfile(source_path, kf_path)
        
    t_sec = round(frame_number / fps, 2)
    title = (
        f"KEYFRAME {keyframe_index + 1} / {len(visual_keyframes)}\n"
        f"Original Frame: {frame_number + 1} / {total_frames}   |   "
        f"Time: {t_sec:.2f} sec   |   "
        f"Sharpness: {sharpness:.2f}"
    )
    
    keyframe_records.append({
        "keyframe_index": keyframe_index,
        "keyframe_number": keyframe_index + 1,
        "frame_number": frame_number + 1,
        "original_frame_number": frame_number + 1,
        "frame_index": frame_number,
        "timestamp": t_sec,
        "timestamp_sec": t_sec,
        "sharpness": round(sharpness, 2),
        "title": title,
        "filename": kf_filename,
        "path": os.path.abspath(kf_path),
        "url": f"/api/keyframes/{keyframe_index}",
        "image_url": f"/api/keyframes/{keyframe_index}"
    })

selected_frames_records = []
for keyframe_index, (frame_number, sharpness) in enumerate(selected_frames):
    t_sec = round(frame_number / fps, 2)
    title = (
        f"KEYFRAME {keyframe_index + 1} / {len(selected_frames)}\n"
        f"Original Frame: {frame_number + 1} / {total_frames}   |   "
        f"Time: {t_sec:.2f} sec   |   "
        f"Sharpness: {sharpness:.2f}"
    )
    selected_frames_records.append({
        "keyframe_index": keyframe_index,
        "keyframe_number": keyframe_index + 1,
        "frame_number": frame_number + 1,
        "original_frame_number": frame_number + 1,
        "frame_index": frame_number,
        "timestamp": t_sec,
        "timestamp_sec": t_sec,
        "sharpness": round(sharpness, 2),
        "title": title,
        "url": f"/api/frames/{frame_number}",
        "image_url": f"/api/frames/{frame_number}"
    })

sharpness_summary = {
    "frames_analyzed": len(frame_numbers),
    "min_score": round(min(blur_scores), 2),
    "max_score": round(max(blur_scores), 2),
    "avg_score": round(float(np.mean(blur_scores)), 2),
    "original_frames": len(frame_numbers),
    "after_blur_filtering": len(selected_frames),
    "before_visual_filtering": len(selected_frames),
    "after_visual_filtering": len(visual_keyframes),
    "removed_redundant_frames": len(selected_frames) - len(visual_keyframes)
}

# Update Cache
with open(CACHE_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

results["keyframes"] = keyframe_records
results["selected_frames"] = selected_frames_records
results["frames"] = frame_items
results["number_of_sharp_frames"] = len(selected_frames)
results["number_of_final_keyframes"] = len(visual_keyframes)
results["sharpness_summary"] = sharpness_summary

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"[COMPLETE] Updated results saved to {CACHE_FILE}. Keyframes saved: {len(keyframe_records)}")
