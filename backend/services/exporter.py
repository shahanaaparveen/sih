"""
Stage 09 - real export files (ROADMAP.md P9.2).

Turns the fused cloud / georeferenced reconstruction into standard deliverables:
  * LAS  - colored point cloud (the problem statement's PLY/LAS requirement)
  * GeoJSON - the camera flight track in real lon/lat, an honest geospatial artifact

PLY is written directly by fusion.py (Open3D). Mesh (OBJ/GLB) is the Phase 9.1 stretch goal.
"""
import json
import os
from typing import Any, Dict

import numpy as np
import open3d as o3d

from services.colmap_io import camera_center, frame_index_from_name, load_model
from services.telemetry import enu_to_geodetic


def export_las(ply_path: str, out_las: str) -> Dict[str, Any]:
    import laspy
    pcd = o3d.io.read_point_cloud(ply_path)
    P = np.asarray(pcd.points)
    if len(P) == 0:
        raise RuntimeError("Point cloud is empty; build the dense cloud first.")
    C = np.asarray(pcd.colors)

    header = laspy.LasHeader(point_format=2, version="1.2")   # format 2 carries RGB
    header.offsets = P.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]                     # mm precision
    las = laspy.LasData(header)
    las.x, las.y, las.z = P[:, 0], P[:, 1], P[:, 2]
    if len(C) == len(P):
        las.red = (np.clip(C[:, 0], 0, 1) * 65535).astype(np.uint16)
        las.green = (np.clip(C[:, 1], 0, 1) * 65535).astype(np.uint16)
        las.blue = (np.clip(C[:, 2], 0, 1) * 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(out_las), exist_ok=True)
    las.write(out_las)
    return {"points": int(len(P)), "las": out_las, "bytes": os.path.getsize(out_las)}


def export_camera_track_geojson(stage_dir: str, out_geojson: str) -> Dict[str, Any]:
    """Camera centres in real lon/lat (needs Stage 07 georef). LineString of the flight + a point per camera."""
    georef_path = os.path.join(stage_dir, "georef.json")
    if not os.path.isfile(georef_path):
        raise RuntimeError("Georeference the project first (Stage 07).")
    with open(georef_path, "r", encoding="utf-8") as f:
        gr = json.load(f)
    tr = gr.get("transform") or {}
    s = float(tr.get("scale"))
    R = np.asarray(tr.get("rotation"), dtype=float)
    t = np.asarray(tr.get("translation"), dtype=float)
    lat0, lon0, alt0 = gr["origin_latlonalt"]

    model = load_model(os.path.join(stage_dir, "results", "model"))
    ims = sorted(model.images.values(), key=lambda im: frame_index_from_name(im.name) or 0)

    features, line = [], []
    for im in ims:
        enu = s * (camera_center(im) @ R.T) + t
        lat, lon, alt = enu_to_geodetic(enu[0], enu[1], enu[2], lat0, lon0, alt0)
        line.append([round(lon, 8), round(lat, 8), round(float(alt), 2)])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8), round(float(alt), 2)]},
            "properties": {"frame_index": frame_index_from_name(im.name), "name": im.name},
        })
    features.insert(0, {"type": "Feature", "geometry": {"type": "LineString", "coordinates": line},
                        "properties": {"role": "flight_track", "gps_source": gr.get("source")}})
    fc = {"type": "FeatureCollection", "features": features,
          "properties": {"gps_source": gr.get("source"), "caveat": gr.get("caveat")}}
    os.makedirs(os.path.dirname(out_geojson), exist_ok=True)
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(fc, f)
    return {"cameras": len(ims), "geojson": out_geojson, "gps_source": gr.get("source")}
