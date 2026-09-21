"""
Measures Stage 04 camera-pose accuracy against the known ground truth of a synthetic flight
(see make_synthetic_flight.py).

COLMAP's world frame and scale are arbitrary, so its camera centres are first aligned to the true
ones with the best similarity transform (Umeyama). What remains is real reconstruction error.

Usage: python scratch/eval_stage04_accuracy.py MODEL_DIR GROUNDTRUTH.json
  MODEL_DIR = folder with cameras.txt / images.txt / points3D.txt   (work/model)
Image names must follow the package convention keyframe_XXXX_original_YYYY.jpg (YYYY = video frame).
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "services"))
from colmap_io import camera_center, load_model, qvec_to_rotmat, similarity_align  # noqa: E402

def evaluate(model_dir, groundtruth_path):
    """Returns a dict of accuracy numbers, or None when fewer than 3 images are registered."""
    model = load_model(model_dir)
    gt = json.load(open(groundtruth_path))
    gt_frames = {f["frame_index"]: f for f in gt["frames"]}

    rows = []
    for im in model.images.values():
        m = re.search(r"original_(\d+)", im.name)
        frame = int(m.group(1))
        truth = gt_frames[frame]
        R_cw = qvec_to_rotmat(im.qvec)
        rows.append((frame, camera_center(im), R_cw.T, np.array(truth["center"]), np.array(truth["R_world_from_cam"])))
    rows.sort(key=lambda r: r[0])
    if len(rows) < 3:
        return None

    est = np.array([r[1] for r in rows])
    ref = np.array([r[3] for r in rows])
    s, R, t = similarity_align(est, ref)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - ref, axis=1)
    path_len = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()

    # orientation: the similarity rotation R maps the estimated world frame onto the true one
    ang = []
    for _, _, Rwc_est, _, Rwc_true in rows:
        d = Rwc_true.T @ (R @ Rwc_est)
        ang.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    ang = np.array(ang)
    cam = next(iter(model.cameras.values()))
    return {
        "registered": len(rows), "scale": s,
        "pos_err_mean_m": float(err.mean()), "pos_err_median_m": float(np.median(err)), "pos_err_max_m": float(err.max()),
        "path_len_m": float(path_len), "pos_err_pct_of_path": float(100 * err.mean() / path_len),
        "ang_err_mean_deg": float(ang.mean()), "ang_err_max_deg": float(ang.max()),
        "focal_est": float(cam.params[0]), "focal_true": float(gt["focal_px"]),
        "focal_err_pct": float(100 * abs(cam.params[0] - gt["focal_px"]) / gt["focal_px"]),
    }


if __name__ == "__main__":
    r = evaluate(sys.argv[1], sys.argv[2])
    if r is None:
        sys.exit("fewer than 3 images registered")
    print(f"registered images        : {r['registered']}")
    print(f"estimated scale factor   : {r['scale']:.4f}  (COLMAP units -> metres)")
    print(f"camera position error    : mean {r['pos_err_mean_m']:.3f} m | median {r['pos_err_median_m']:.3f} m | max {r['pos_err_max_m']:.3f} m")
    print(f"   relative to path      : {r['pos_err_pct_of_path']:.3f}% of the {r['path_len_m']:.1f} m flown")
    print(f"camera orientation error : mean {r['ang_err_mean_deg']:.3f} deg | max {r['ang_err_max_deg']:.3f} deg")
    print(f"focal length             : estimated {r['focal_est']:.2f} px vs true {r['focal_true']:.2f} px ({r['focal_err_pct']:.2f}% off)")
