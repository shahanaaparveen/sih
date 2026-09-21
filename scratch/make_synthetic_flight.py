"""
Renders a synthetic drone flight over a textured 3D terrain (hills + buildings) and writes
  <out>.mp4              the video
  <out>_groundtruth.json exact camera pose of every frame (world_from_cam rotation + centre, metres)

Purpose: give the Stage 04 (COLMAP) pipeline footage with real parallax AND known camera
positions, so reconstruction accuracy can be measured instead of just "it ran".

World frame: X east, Y north, Z up (metres). Camera frame: OpenCV (x right, y down, z forward).

Usage: python scratch/make_synthetic_flight.py OUT_PREFIX [frames=120] [width=800] [height=450]
"""
import json
import sys

import cv2
import numpy as np

out_prefix = sys.argv[1]
n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 120
W = int(sys.argv[3]) if len(sys.argv) > 3 else 800
H = int(sys.argv[4]) if len(sys.argv) > 4 else 450
FOCAL = 0.8125 * W          # ~63 deg horizontal field of view
RES = 0.25                  # heightfield cell size in metres
HALF = 130.0                # terrain spans [-HALF, HALF] in X and Y

rng = np.random.default_rng(7)


def smooth_noise(size, cells, seed):
    r = np.random.default_rng(seed).random((cells, cells)).astype(np.float32)
    return cv2.resize(r, (size, size), interpolation=cv2.INTER_CUBIC)


# ---- terrain heightfield: rolling hills + boxy buildings ---------------------------------
N = int(2 * HALF / RES)
height = 12.0 * smooth_noise(N, 6, 1) + 4.0 * smooth_noise(N, 20, 2)
for _ in range(70):
    cx, cy = rng.uniform(-100, 100, 2)
    sx, sy = rng.uniform(6, 16, 2)
    top = rng.uniform(8, 26)
    x0, x1 = int((cx - sx + HALF) / RES), int((cx + sx + HALF) / RES)
    y0, y1 = int((cy - sy + HALF) / RES), int((cy + sy + HALF) / RES)
    height[y0:y1, x0:x1] = height[y0:y1, x0:x1].mean() + top

# ---- procedural texture (multi-scale noise so SIFT has blobs and corners at every scale) -
T = 2048
tex = np.zeros((T, T, 3), np.float32)
for octave, amp in [(8, 0.35), (32, 0.3), (128, 0.25), (512, 0.2)]:
    for c in range(3):
        tex[..., c] += amp * smooth_noise(T, octave, 100 * octave + c)
tex = np.clip((tex - tex.min()) / (tex.max() - tex.min()), 0, 1)

# ---- fixed sun lighting from the heightfield normals (view independent) -------------------
gy, gx = np.gradient(height, RES)
normal = np.dstack([-gx, -gy, np.ones_like(gx)])
normal /= np.linalg.norm(normal, axis=2, keepdims=True)
sun = np.array([0.4, -0.3, 0.85])
sun /= np.linalg.norm(sun)
shade = np.clip(normal @ sun, 0, 1).astype(np.float32)


def cell(xy):
    ix = np.clip(((xy[:, 0] + HALF) / RES).astype(np.int32), 0, N - 1)
    iy = np.clip(((xy[:, 1] + HALF) / RES).astype(np.int32), 0, N - 1)
    return ix, iy


def ground_z(xy):
    ix, iy = cell(xy)
    return height[iy, ix]


# ---- camera path -------------------------------------------------------------------------
def camera_pose(t):
    cx = -45.0 + 90.0 * t
    cy = 25.0 * np.sin(2 * np.pi * t * 0.75)
    cz = 70.0 + 6.0 * np.sin(2 * np.pi * t * 1.3)
    dx = 90.0
    dy = 25.0 * 2 * np.pi * 0.75 * np.cos(2 * np.pi * t * 0.75)
    yaw = np.arctan2(dy, dx) + np.deg2rad(12) * np.sin(2 * np.pi * t * 1.1)
    pitch = np.deg2rad(58 + 6 * np.sin(2 * np.pi * t * 0.9))  # degrees below the horizon
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), -np.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_wc = np.stack([right, down, fwd], axis=1)  # columns = camera axes in world
    return np.array([cx, cy, cz]), R_wc


uu, vv = np.meshgrid(np.arange(W), np.arange(H))
cam_rays = np.stack([(uu.ravel() - W / 2) / FOCAL, (vv.ravel() - H / 2) / FOCAL, np.ones(W * H)], axis=1)


def render(origin, R_wc):
    d = cam_rays @ R_wc.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = d.shape[0]
    t_hit = np.full(n, np.nan)
    active = np.arange(n)
    t_prev = 0.0
    for t in np.arange(1.0, 220.0, 0.5):
        p = origin + t * d[active]
        below = p[:, 2] <= ground_z(p[:, :2])
        if below.any():
            hit_idx = active[below]
            lo, hi = np.full(hit_idx.shape, t_prev), np.full(hit_idx.shape, t)
            for _ in range(8):  # bisection refine
                mid = 0.5 * (lo + hi)
                pm = origin + mid[:, None] * d[hit_idx]
                under = pm[:, 2] <= ground_z(pm[:, :2])
                hi = np.where(under, mid, hi)
                lo = np.where(under, lo, mid)
            t_hit[hit_idx] = hi
            active = active[~below]
            if active.size == 0:
                break
        t_prev = t
    img = np.zeros((n, 3), np.float32)
    sky = np.isnan(t_hit)
    ok = ~sky
    p = origin + t_hit[ok, None] * d[ok]
    ix, iy = cell(p[:, :2])
    tx = ((p[:, 0] + HALF) / (2 * HALF) * (T - 1)).astype(np.int32).clip(0, T - 1)
    ty = ((p[:, 1] + HALF) / (2 * HALF) * (T - 1)).astype(np.int32).clip(0, T - 1)
    img[ok] = tex[ty, tx] * (0.55 + 0.45 * shade[iy, ix])[:, None]
    img[sky] = [0.75, 0.85, 0.95]
    return (img.reshape(H, W, 3)[..., ::-1] * 255).astype(np.uint8)  # RGB -> BGR for OpenCV


writer = cv2.VideoWriter(out_prefix + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
truth = []
for i in range(n_frames):
    t = i / max(1, n_frames - 1)
    origin, R_wc = camera_pose(t)
    writer.write(render(origin, R_wc))
    truth.append({"frame_index": i, "center": origin.tolist(), "R_world_from_cam": R_wc.tolist()})
    if i % 10 == 0:
        print(f"rendered {i + 1}/{n_frames}", flush=True)
writer.release()

json.dump({"width": W, "height": H, "focal_px": FOCAL, "frames": truth},
          open(out_prefix + "_groundtruth.json", "w"))
print("done", out_prefix)
