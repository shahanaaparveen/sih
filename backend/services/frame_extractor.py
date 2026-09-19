import os
import cv2
from typing import List

def extract_all_frames(video_path: str, output_dir: str) -> List[str]:
    """
    Extracts every single frame from the video and saves to output_dir.
    Matches tested Colab pipeline logic.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Remove previous frames in output_dir
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_paths = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = f"frame_{frame_id:04d}.jpg"
        frame_path = os.path.join(output_dir, filename)

        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_id += 1

    cap.release()
    print(f"[FRAME_EXTRACTOR] Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths
