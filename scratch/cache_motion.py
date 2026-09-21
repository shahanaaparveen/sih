"""
Caches the per-frame quantities the SfM keyframe selector consumes (frame-to-frame similarity transform at
work resolution + Laplacian sharpness), computed exactly like backend/services/video_processor.py does, so the
selector can be tuned on a real video in seconds instead of re-running the whole pipeline.

    python scratch/cache_motion.py VIDEO OUT.npz
"""
import sys

import cv2
import numpy as np

video, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W0, H0 = int(cap.get(3)), int(cap.get(4))
ww = 960 if W0 > 960 else W0
wh = int(H0 * (ww / W0))
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
transforms, sharp, prev = [np.full((2, 3), np.nan)], [], (None, None)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    small = cv2.resize(gray, (ww, wh)) if ww != W0 else gray
    kp, des = orb.detectAndCompute(small, None)
    if i > 0:
        M = None
        if prev[1] is not None and des is not None:
            try:
                m = sorted(bf.match(prev[1], des), key=lambda x: x.distance)[:80]
                if len(m) >= 8:
                    p1 = np.float32([prev[0][x.queryIdx].pt for x in m])
                    p2 = np.float32([kp[x.trainIdx].pt for x in m])
                    M, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC)
            except Exception:
                M = None
        transforms.append(np.full((2, 3), np.nan) if M is None else M)
    prev = (kp, des)
    i += 1
    if i % 300 == 0:
        print(f"{i}/{n}", flush=True)
np.savez(out, transforms=np.array(transforms), sharpness=np.array(sharp), width=ww, height=wh, fps=cap.get(5))
print("saved", out, len(sharp), "frames")
