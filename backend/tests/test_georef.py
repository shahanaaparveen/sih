import csv
import math
import os
import sys
import tempfile
import unittest

import numpy as np

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from services import georef  # noqa: E402
from services.telemetry import enu_to_geodetic, geodetic_to_enu  # noqa: E402


def _write_model(stage_dir, centers):
    md = os.path.join(stage_dir, "results", "model")
    os.makedirs(md)
    with open(os.path.join(md, "cameras.txt"), "w", encoding="utf-8") as f:
        f.write("1 SIMPLE_RADIAL 1600 900 1100 800 450 0.0\n")
    with open(os.path.join(md, "images.txt"), "w", encoding="utf-8") as f:
        for i, c in enumerate(centers):
            t = -np.asarray(c, dtype=float)              # R = I  ->  t = -C  ->  camera_center = C
            f.write(f"{i+1} 1 0 0 0 {t[0]} {t[1]} {t[2]} 1 keyframe_{i:04d}_original_{i*10:04d}.jpg\n\n")
    open(os.path.join(md, "points3D.txt"), "w").close()


class TestEnuRoundTrip(unittest.TestCase):
    def test_geodetic_enu_round_trip(self):
        lat0, lon0, alt0 = 12.9716, 77.5946, 915.0
        for e, n, u in [(0, 0, 0), (120.0, -80.0, 15.0), (-300.0, 200.0, -5.0)]:
            la, lo, al = enu_to_geodetic(e, n, u, lat0, lon0, alt0)
            e2, n2, u2 = geodetic_to_enu(la, lo, al, lat0, lon0, alt0)
            self.assertAlmostEqual(e, e2, places=3)
            self.assertAlmostEqual(n, n2, places=3)
            self.assertAlmostEqual(u, u2, places=6)


class TestGeoreference(unittest.TestCase):
    def test_recovers_scale_and_near_zero_rmse_on_exact_gps(self):
        centers = [(i, 0.5 * math.sin(i), 0.3 * i) for i in range(8)]   # non-degenerate track
        stage = tempfile.mkdtemp()
        _write_model(stage, centers)

        S, lat0, lon0, alt0 = 6.5, 12.9, 77.6, 900.0
        enu = S * np.asarray(centers)
        enu -= enu.mean(0)
        csv_path = os.path.join(stage, "gps.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "lat", "lon", "alt"])
            for i, (e, n, u) in enumerate(enu):
                la, lo, al = enu_to_geodetic(e, n, u, lat0, lon0, alt0)
                w.writerow([i * 10, la, lo, al])

        rep = georef.georeference(stage, {"duration": 10.0, "filename": "x.mp4"},
                                  project_gps_csv=csv_path, transform_cloud=False)
        self.assertEqual(rep["source"], "real")
        self.assertAlmostEqual(rep["scale_units_to_m"], S, delta=0.05)   # scale recovered
        self.assertLess(rep["rmse_3d_m"], 0.05)                          # exact data -> ~0 (ENU approx only)
        self.assertLess(rep["rmse_horizontal_m"], 0.05)

    def test_too_few_gps_frames_raises(self):
        stage = tempfile.mkdtemp()
        _write_model(stage, [(0, 0, 0), (1, 0, 0)])   # only 2 frames
        with self.assertRaises(RuntimeError):
            georef.georeference(stage, {"duration": 5.0, "filename": "x.mp4"}, transform_cloud=False)


if __name__ == "__main__":
    unittest.main()
