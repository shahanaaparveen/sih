import cv2
import time
import numpy as np

video_path = 'assets/media/sample_4k.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Testing on {video_path}: {total_frames} frames, {w}x{h} @ {fps}fps")

t0 = time.time()

# ORB detector on downscaled grayscale
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

work_w = 960
work_h = int(h * (work_w / w))
scale_factor = w / work_w

trajectory = [{"x": 0.0, "y": 0.0}]
blur_scores = []
frame_id = 0
prev_kp = None
prev_des = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Downscaled grayscale for high-speed ML processing
    small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (work_w, work_h))

    # 1. Laplacian sharpness score
    sharpness = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())
    blur_scores.append(sharpness)

    # 2. ORB feature detection & trajectory
    curr_kp, curr_des = orb.detectAndCompute(small_gray, None)

    if prev_kp is None or prev_des is None or curr_des is None:
        if frame_id > 0:
            trajectory.append(dict(trajectory[-1]))
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
                    new_x = trajectory[-1]["x"] + dx
                    new_y = trajectory[-1]["y"] + dy
                    trajectory.append({"x": round(new_x, 2), "y": round(new_y, 2)})
                else:
                    trajectory.append(dict(trajectory[-1]))
            else:
                trajectory.append(dict(trajectory[-1]))
        except Exception:
            trajectory.append(dict(trajectory[-1]))

    prev_kp = curr_kp
    prev_des = curr_des
    frame_id += 1

    if frame_id % 200 == 0:
        print(f"Processed {frame_id}/{total_frames} frames in {time.time() - t0:.2f}s...")

cap.release()
total_time = time.time() - t0
print(f"DONE! Processed ALL {frame_id} frames in {total_time:.2f} seconds ({total_time/frame_id*1000:.1f} ms/frame).")
print(f"Trajectory points: {len(trajectory)}, Sharpness scores: {len(blur_scores)}")
