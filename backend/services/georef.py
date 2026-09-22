"""
Stage 07 - georeferencing (ROADMAP.md P7).

COLMAP reconstructs only up to an arbitrary similarity (scale, rotation, position). This aligns the
reconstruction to the flight's GPS track with a robust similarity fit (Umeyama, similarity_align),
turning COLMAP units into metres in a local ENU frame, and reports horizontal / vertical / 3D RMSE on
HELD-OUT frames (an honest accuracy estimate, not a fit residual).

With real GPS/RTK this is a genuine metric-accuracy figure. With simulated GPS it validates the
pipeline only, and the report says so.
"""
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

from services.colmap_io import camera_center, frame_index_from_name, load_model, similarity_align
from services.telemetry import geodetic_to_enu, get_gps_track


def _rmse(a: np.ndarray) -> Optional[float]:
    return round(float(np.sqrt(np.mean(a ** 2))), 3) if len(a) else None


def georeference(stage_dir: str, results: Dict[str, Any], project_gps_csv: Optional[str] = None,
                 holdout_every: int = 4, transform_cloud: bool = True) -> Dict[str, Any]:
    model_dir = os.path.join(stage_dir, "results", "model")
    if not os.path.isfile(os.path.join(model_dir, "images.txt")):
        raise FileNotFoundError("No Stage 04 model; run pose+depth first.")
    model = load_model(model_dir)
    track, source = get_gps_track(results, model, project_gps_csv)

    ims = sorted(model.images.values(), key=lambda im: frame_index_from_name(im.name) or 0)
    src, geo = [], []
    for im in ims:
        fi = frame_index_from_name(im.name)
        if fi in track:
            src.append(camera_center(im))
            geo.append(track[fi])
    if len(src) < 4:
        raise RuntimeError(f"Need at least 4 registered frames with GPS to georeference (have {len(src)}).")
    src = np.asarray(src, dtype=np.float64)
    geo = np.asarray(geo, dtype=np.float64)

    lat0, lon0, alt0 = float(geo[:, 0].mean()), float(geo[:, 1].mean()), float(geo[:, 2].mean())
    dst = np.array([geodetic_to_enu(la, lo, al, lat0, lon0, alt0) for la, lo, al in geo])

    n = len(src)
    hold = np.zeros(n, dtype=bool)
    hold[::holdout_every] = True
    if hold.all() or (~hold).sum() < 3:      # too few to hold any out: fit on all, evaluate on all
        hold[:] = False
    s, R, t = similarity_align(src[~hold] if hold.any() else src, dst[~hold] if hold.any() else dst)

    pred = s * (src @ R.T) + t                # metric ENU predicted from the reconstruction
    resid = pred - dst
    ev = hold if hold.any() else np.ones(n, dtype=bool)   # evaluate on held-out frames (or all if none held out)

    report: Dict[str, Any] = {
        "source": source,
        "gps_frames": n,
        "fit_frames": int((~hold).sum()) if hold.any() else n,
        "holdout_frames": int(hold.sum()),
        "scale_units_to_m": round(float(s), 6),
        "origin_latlonalt": [round(lat0, 7), round(lon0, 7), round(alt0, 2)],
        "rmse_horizontal_m": _rmse(np.linalg.norm(resid[ev][:, :2], axis=1)),
        "rmse_vertical_m": _rmse(np.abs(resid[ev][:, 2])),
        "rmse_3d_m": _rmse(np.linalg.norm(resid[ev], axis=1)),
        "evaluated_on": "holdout" if hold.any() else "all_frames",
        # full similarity (COLMAP units -> metric ENU): metric = scale * (p @ R^T) + t. Lets exports
        # (metric cloud, GeoJSON track) reuse the exact alignment without refitting.
        "transform": {"scale": float(s), "rotation": R.tolist(), "translation": t.tolist()},
    }
    if source == "simulated":
        report["caveat"] = ("GPS is SIMULATED: these metric figures validate the georeferencing pipeline "
                            "only. Real accuracy needs the flight's actual GPS/RTK track (supply a "
                            "frame_index,lat,lon,alt CSV).")

    if transform_cloud:
        ply = os.path.join(stage_dir, "dense.ply")
        if os.path.isfile(ply):
            pcd = o3d.io.read_point_cloud(ply)
            P = np.asarray(pcd.points)
            pcd.points = o3d.utility.Vector3dVector(s * (P @ R.T) + t)   # -> metric ENU
            out = os.path.join(stage_dir, "dense_metric.ply")
            o3d.io.write_point_cloud(out, pcd, write_ascii=False)
            report["metric_ply"] = out
            report["metric_ply_points"] = int(len(P))

    with open(os.path.join(stage_dir, "georef.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
