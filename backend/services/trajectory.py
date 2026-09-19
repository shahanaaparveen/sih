import os
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

def calculate_camera_trajectory(
    frame_paths: List[str],
    output_json_path: str = None
) -> Tuple[List[Dict[str, float]], int, int]:
    """
    Calculates 2D camera trajectory using ALL extracted frames.
    Produces exactly one trajectory point per frame (e.g. 712 frames -> 712 points).
    Matches Colab notebook Cell 17.
    """
    if not frame_paths:
        return [], 0, 0

    if len(frame_paths) == 1:
        single_point = [{"x": 0.0, "y": 0.0}]
        return single_point, 0, 0

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Initial trajectory point at (0.0, 0.0)
    trajectory: List[Dict[str, float]] = [{"x": 0.0, "y": 0.0}]

    first_frame = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame for trajectory: {frame_paths[0]}")

    kp1, des1 = orb.detectAndCompute(first_frame, None)

    successful = 0
    failed = 0

    for i in range(1, len(frame_paths)):
        curr = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
        if curr is None:
            trajectory.append(dict(trajectory[-1]))
            failed += 1
            continue

        kp2, des2 = orb.detectAndCompute(curr, None)

        if des1 is None or des2 is None:
            trajectory.append(dict(trajectory[-1]))
            failed += 1
            continue

        try:
            matches = bf.match(des1, des2)
        except Exception:
            trajectory.append(dict(trajectory[-1]))
            failed += 1
            continue

        matches = sorted(matches, key=lambda m: m.distance)
        good_matches = matches[:100]

        if len(good_matches) < 8:
            trajectory.append(dict(trajectory[-1]))
            failed += 1
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

        if M is None:
            trajectory.append(dict(trajectory[-1]))
            failed += 1
            continue

        dx = float(M[0, 2])
        dy = float(M[1, 2])

        prev = trajectory[-1]
        new_x = float(prev["x"] + dx)
        new_y = float(prev["y"] + dy)

        trajectory.append({"x": round(new_x, 3), "y": round(new_y, 3)})
        successful += 1

        kp1 = kp2
        des1 = des2

        if i % 50 == 0:
            print(f"[TRAJECTORY] Processed {i}/{len(frame_paths)} frames...")

    print(f"[TRAJECTORY] Complete. Points: {len(trajectory)}, Successful: {successful}, Failed: {failed}")

    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_frames": len(frame_paths),
                "trajectory_points": len(trajectory),
                "successful_matches": successful,
                "failed_frames": failed,
                "points": trajectory
            }, f, indent=2)

    return trajectory, successful, failed
