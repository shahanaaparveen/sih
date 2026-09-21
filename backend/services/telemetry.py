"""
Telemetry / GPS for georeferencing (ROADMAP.md P3.2 + P7 input).

Provides a per-video-frame GPS track {frame_index: (lat, lon, alt)}:
  * real   - parsed from a project CSV (frame_index,lat,lon,alt). GPX/SRT can be added later.
  * simulated - a plausible track derived from the reconstruction's own camera path, with realistic
                consumer-GPS noise, so the georeferencing pipeline can be exercised before real data
                arrives. ALWAYS labelled "simulated"; never presented as metric truth.

Also holds the local-tangent-plane (ENU) conversions used by georeferencing.
"""
import csv
import math
import os
from typing import Dict, Optional, Tuple

import numpy as np

from services.colmap_io import camera_center, frame_index_from_name

EARTH_R = 6378137.0   # WGS-84 mean radius (m)


def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0) -> Tuple[float, float, float]:
    """Small-area equirectangular local tangent plane about (lat0, lon0, alt0). Good to ~cm over a few km."""
    e = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * EARTH_R
    return e, n, alt - alt0


def enu_to_geodetic(e, n, u, lat0, lon0, alt0) -> Tuple[float, float, float]:
    lat = lat0 + math.degrees(n / EARTH_R)
    lon = lon0 + math.degrees(e / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon, alt0 + u


def parse_gps_csv(path: str) -> Dict[int, Tuple[float, float, float]]:
    """CSV with a header including frame_index (or frame), lat, lon, and optional alt."""
    out: Dict[int, Tuple[float, float, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                fi = int(float(row.get("frame_index", row.get("frame"))))
                out[fi] = (float(row["lat"]), float(row["lon"]), float(row.get("alt", 0.0) or 0.0))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def simulate_gps(model, duration_s: float = 20.0, speed_mps: float = 5.0,
                 origin: Tuple[float, float, float] = (12.9716, 77.5946, 915.0),
                 hnoise_m: float = 1.5, vnoise_m: float = 3.0, seed: int = 0
                 ) -> Dict[int, Tuple[float, float, float]]:
    """
    A plausible GPS track built from the reconstruction's own camera centres: pick a real-world scale
    from an assumed cruise speed x duration, apply a random heading, add realistic GPS noise, and place
    it at `origin`. This exercises the georeferencing pipeline; it is NOT real positioning data.
    """
    ims = sorted(model.images.values(), key=lambda im: frame_index_from_name(im.name) or 0)
    frames = [frame_index_from_name(im.name) for im in ims]
    centers = np.array([camera_center(im) for im in ims])
    if len(centers) < 2:
        return {}
    path_units = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()) or 1.0
    scale = max(speed_mps * max(duration_s, 1.0), 10.0) / path_units      # units -> metres
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * math.pi)
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    enu = scale * (centers @ Rz.T)
    enu -= enu.mean(0)
    enu += rng.normal(0, [hnoise_m, hnoise_m, vnoise_m], enu.shape)
    lat0, lon0, alt0 = origin
    return {fi: enu_to_geodetic(e, n, u, lat0, lon0, alt0)
            for fi, (e, n, u) in zip(frames, enu) if fi is not None}


def get_gps_track(results: dict, model, project_gps_csv: Optional[str] = None):
    """Returns (track {frame_index:(lat,lon,alt)}, source 'real'|'simulated')."""
    if project_gps_csv and os.path.isfile(project_gps_csv):
        track = parse_gps_csv(project_gps_csv)
        if track:
            return track, "real"
    return simulate_gps(model, duration_s=float(results.get("duration") or 20.0)), "simulated"
