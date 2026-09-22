"""
Stage 08 - confidence & coverage map (ROADMAP.md P8).

Single-pass reconstruction is uneven: surfaces seen from many viewpoints are well constrained, while
those glimpsed once (or never) are not. This turns that into an explicit, honest signal instead of a
uniformly "pretty" cloud.

For every dense point we count how many camera views actually observe it (project into each camera; in
front and inside the image). Viewpoint coverage is the dominant reliability cue for multi-view geometry,
so confidence = coverage normalised to `full_views`, and the report exposes the coverage distribution
(the "coverage-gap" insight: which fraction of the model is weakly observed).

Outputs (per project, in the Stage 04 dir):
  dense_confidence.ply   dense cloud recoloured by confidence (turbo: red = low, green/blue = high)
  confidence.json        coverage stats + view-count histogram + depth reliability
"""
import json
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import open3d as o3d

from services.colmap_io import load_model, qvec_to_rotmat
from services.fusion import _intrinsics


def compute_confidence(stage_dir: str, out_ply: Optional[str] = None, full_views: int = 4,
                       min_views: int = 2, max_points: int = 300000) -> Dict[str, Any]:
    results = os.path.join(stage_dir, "results")
    model_dir = os.path.join(results, "model")
    ply = os.path.join(stage_dir, "dense.ply")
    if not os.path.isfile(os.path.join(model_dir, "images.txt")):
        raise FileNotFoundError("No Stage 04 model; run pose+depth first.")
    if not os.path.isfile(ply):
        raise FileNotFoundError("Build the dense cloud first.")

    model = load_model(model_dir)
    pcd = o3d.io.read_point_cloud(ply)
    P = np.asarray(pcd.points)
    if len(P) == 0:
        raise RuntimeError("Dense cloud is empty.")
    if len(P) > max_points:
        idx = np.sort(np.random.default_rng(0).choice(len(P), max_points, replace=False))
        P = P[idx]
    N = len(P)

    # Count, per point, how many cameras see it (in front + inside the image rectangle).
    view_count = np.zeros(N, dtype=np.int32)
    for im in model.images.values():
        R = qvec_to_rotmat(im.qvec)
        cam = model.cameras[im.camera_id]
        fx, fy, cx, cy = _intrinsics(cam)
        Xc = P @ R.T + im.tvec                      # world -> camera
        z = Xc[:, 2]
        safe = z > 1e-6
        u = np.where(safe, fx * Xc[:, 0] / np.where(safe, z, 1) + cx, -1)
        v = np.where(safe, fy * Xc[:, 1] / np.where(safe, z, 1) + cy, -1)
        view_count += (safe & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)).astype(np.int32)

    conf = np.clip(view_count / float(full_views), 0.0, 1.0)
    max_v = int(view_count.max()) if N else 0
    report: Dict[str, Any] = {
        "points": int(N),
        "mean_views": round(float(view_count.mean()), 3),
        "median_views": float(np.median(view_count)),
        "max_views": max_v,
        "min_views_threshold": min_views,
        "full_views": full_views,
        "well_observed_fraction": round(float(np.mean(view_count >= min_views)), 4),
        "single_view_fraction": round(float(np.mean(view_count <= 1)), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "view_histogram": {str(k): int(np.sum(view_count == k)) for k in range(0, max_v + 1)},
    }
    dr = os.path.join(results, "depth_report.json")
    if os.path.isfile(dr):
        with open(dr, "r", encoding="utf-8") as f:
            report["depth_holdout_rel_err_median"] = json.load(f).get("holdout_rel_err_median")

    # Recolour the cloud by confidence (turbo: warm/red = weakly observed, cool = well observed).
    turbo = cv2.applyColorMap((conf * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    rgb = turbo[:, 0, ::-1].astype(np.float64) / 255.0     # BGR -> RGB
    cpcd = o3d.geometry.PointCloud()
    cpcd.points = o3d.utility.Vector3dVector(P)
    cpcd.colors = o3d.utility.Vector3dVector(rgb)
    out_ply = out_ply or os.path.join(stage_dir, "dense_confidence.ply")
    o3d.io.write_point_cloud(out_ply, cpcd, write_ascii=False)
    report["confidence_ply"] = out_ply

    with open(os.path.join(stage_dir, "confidence.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
