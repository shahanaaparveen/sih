"""
Stage 05/06 - dense point cloud (ROADMAP.md P6).

Back-projects every aligned depth map into world space using its COLMAP pose + intrinsics, colours each
point from the source image, then fuses and cleans the result with Open3D into one dense coloured cloud.

Depth maps (from depth_pipeline.py) are z in the camera frame, in COLMAP's arbitrary world units, so the
fused cloud lives in the same frame as the sparse model and camera poses. Metric scale comes later
(georeferencing, Stage 07).

    stats = build_dense_cloud(stage_dir, keyframes_dir, frames_dir, out_ply)
"""
import json
import os
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import open3d as o3d

from services.colmap_io import camera_center, frame_index_from_name, load_model, qvec_to_rotmat

ProgressCB = Optional[Callable[[str, str], None]]


def _rgb_path(name: str, keyframes_dir: str, frames_dir: str) -> Optional[str]:
    """Source image for a registered COLMAP image: a keyframe (keyframe_*) or a gap-fill video frame (fill_*)."""
    direct = os.path.join(keyframes_dir, os.path.basename(name))
    if os.path.isfile(direct):
        return direct
    fi = frame_index_from_name(name)
    if fi is not None:
        cand = os.path.join(frames_dir, f"frame_{fi:04d}.jpg")
        if os.path.isfile(cand):
            return cand
    return None


def _intrinsics(cam) -> tuple:
    """(fx, fy, cx, cy) for the camera at its own width/height. Radial distortion is ignored (small at
    the 0.5px reprojection error we see; can be undistorted later for edge accuracy)."""
    p = cam.params
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    # PINHOLE, OPENCV, FULL_OPENCV, ... -> fx, fy, cx, cy
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def build_dense_cloud(stage_dir: str, keyframes_dir: str, frames_dir: str, out_ply: str,
                      pixel_stride: int = 2, voxel_size: float = 0.0, max_points: int = 3_000_000,
                      masks_dir: Optional[str] = None, progress: ProgressCB = None) -> Dict[str, Any]:
    results_dir = os.path.join(stage_dir, "results")
    model_dir = os.path.join(results_dir, "model")
    depth_dir = os.path.join(results_dir, "depth")
    if masks_dir and not os.path.isdir(masks_dir):
        masks_dir = None
    if not os.path.isfile(os.path.join(model_dir, "images.txt")):
        raise FileNotFoundError("No Stage 04 model found; run pose+depth first.")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError("No depth maps found; run Stage 04 with depth enabled.")

    model = load_model(model_dir)
    z_far_by_name = {}
    rep_path = os.path.join(results_dir, "depth_report.json")
    if os.path.isfile(rep_path):
        with open(rep_path, "r", encoding="utf-8") as f:
            z_far_by_name = {e["name"]: e.get("z_far") for e in json.load(f).get("images", [])}

    images = sorted(model.images.values(), key=lambda im: frame_index_from_name(im.name) or 0)
    pts_chunks, col_chunks, used = [], [], 0
    for n, im in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(im.name))[0]
        npy = os.path.join(depth_dir, stem + ".npy")
        rgb_path = _rgb_path(im.name, keyframes_dir, frames_dir)
        if not os.path.isfile(npy) or not rgb_path:
            continue
        depth = np.load(npy).astype(np.float32)
        sh, sw = depth.shape
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

        mask = None
        if masks_dir:
            mm = cv2.imread(os.path.join(masks_dir, os.path.basename(im.name) + ".png"), cv2.IMREAD_GRAYSCALE)
            if mm is not None:
                mask = cv2.resize(mm, (sw, sh), interpolation=cv2.INTER_NEAREST)   # 0 = dynamic, drop it

        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = _intrinsics(cam)
        sx, sy = sw / cam.width, sh / cam.height     # depth stored smaller than the COLMAP image
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy

        us = np.arange(0, sw, pixel_stride)
        vs = np.arange(0, sh, pixel_stride)
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu]
        valid = np.isfinite(z) & (z > 0)
        z_far = z_far_by_name.get(im.name)
        if z_far:
            valid &= z < 0.98 * z_far                # drop pixels pinned at the far clamp (sky/background)
        if mask is not None:
            valid &= mask[vv, uu] > 0                # drop dynamic-object pixels (people/vehicles/animals)
        if not valid.any():
            continue
        uu, vv, z = uu[valid], vv[valid], z[valid]

        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z
        cam_pts = np.stack([x, y, z], axis=1)        # camera frame
        R = qvec_to_rotmat(im.qvec)                  # world -> camera
        world = cam_pts @ R + camera_center(im)      # X_world = R^T X_cam + C  (row-vector form)

        pts_chunks.append(world.astype(np.float32))
        col_chunks.append(rgb[vv, uu])
        used += 1
        if progress:
            progress("fuse", f"back-projected {n}/{len(images)} images")

    if not pts_chunks:
        raise RuntimeError("No depth maps could be fused (missing depth or source images).")

    P = np.concatenate(pts_chunks)
    C = np.concatenate(col_chunks)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((C.astype(np.float64) / 255.0))

    if voxel_size <= 0:                              # auto: ~1/600 of the scene extent
        extent = float(np.linalg.norm(P.max(0) - P.min(0)))
        voxel_size = max(extent / 600.0, 1e-6)
    if progress:
        progress("clean", f"downsampling (voxel {voxel_size:.4g}) and removing outliers")
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) > max_points:
        keep = np.random.default_rng(0).choice(len(pcd.points), max_points, replace=False)
        pcd = pcd.select_by_index(np.sort(keep))

    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False)

    pts = np.asarray(pcd.points)
    return {
        "ply": out_ply,
        "points": int(len(pts)),
        "images_fused": used,
        "voxel_size": round(float(voxel_size), 6),
        "bounds_min": [round(float(v), 4) for v in pts.min(0)] if len(pts) else None,
        "bounds_max": [round(float(v), 4) for v in pts.max(0)] if len(pts) else None,
        "bytes": os.path.getsize(out_ply),
    }
